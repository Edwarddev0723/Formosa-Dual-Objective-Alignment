"""formosa_dual.training.checkpoint — save / load training checkpoints.

Checkpoint layout (§5.20):

    outputs/<run_name>/checkpoint-<step>/
    ├── adapter_model.safetensors   # PEFT LoRA weights
    ├── adapter_config.json
    ├── aux_modules.safetensors     # pooler + proj_head + tag_projector.projector
    ├── optimizer.pt
    ├── scheduler.pt
    ├── rng_state.pt
    ├── training_state.json
    └── run_config.yaml
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    accelerator,
    cfg,
    step: int,
    epoch: int,
    best_metric: float | None,
    output_dir: Path,
) -> Path:
    """Persist checkpoint to *output_dir/checkpoint-{step}/*.

    Args:
        model: :class:`~formosa_dual.models.dual_model.DualObjectiveModel`.
        optimizer: PyTorch optimizer (or Accelerator-wrapped).
        scheduler: LR scheduler (or None).
        accelerator: ``accelerate.Accelerator`` instance.
        cfg: :class:`~formosa_dual.config.schema.RunConfig`.
        step: Current global step.
        epoch: Current epoch (0-indexed).
        best_metric: Current best validation metric (or None).
        output_dir: Base output directory.

    Returns:
        Path to the checkpoint directory.
    """
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # PEFT adapter (LoRA)
    backbone = model.backbone
    if hasattr(backbone, "save_pretrained"):
        backbone.save_pretrained(str(ckpt_dir))
        logger.info("Saved PEFT adapter to %s", ckpt_dir)

    # Aux modules (pooler, proj_head, tag_projector.projector)
    aux_state: dict = {}
    if model.pooler is not None:
        aux_state["pooler"] = model.pooler.state_dict()
    if model.proj_head is not None:
        aux_state["proj_head"] = model.proj_head.state_dict()
    if model.tag_projector is not None:
        aux_state["tag_projector_projector"] = model.tag_projector.projector.state_dict()

    if aux_state:
        try:
            from safetensors.torch import save_file as st_save
            # safetensors requires flat {str: Tensor} — prefix keys with submodule
            flat: dict[str, torch.Tensor] = {}
            for prefix, sd in aux_state.items():
                for k, v in sd.items():
                    flat[f"{prefix}.{k}"] = v.contiguous()
            st_save(flat, str(ckpt_dir / "aux_modules.safetensors"))
        except ImportError:
            torch.save(aux_state, ckpt_dir / "aux_modules.pt")
        logger.info("Saved aux modules to %s", ckpt_dir)

    # Optimizer + scheduler
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

    # RNG state
    rng_state = {
        "cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng_state, ckpt_dir / "rng_state.pt")

    # Training state JSON
    training_state = {
        "step": step,
        "epoch": epoch,
        "best_metric": best_metric,
    }
    (ckpt_dir / "training_state.json").write_text(
        json.dumps(training_state, indent=2), encoding="utf-8"
    )

    # Resolved config YAML
    try:
        cfg_dict = cfg.model_dump() if hasattr(cfg, "model_dump") else {}
        (ckpt_dir / "run_config.yaml").write_text(
            yaml.safe_dump(cfg_dict, allow_unicode=True), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("Could not serialize run config to YAML: %s", exc)

    logger.info("Checkpoint saved: %s (step=%d, epoch=%d)", ckpt_dir, step, epoch)
    return ckpt_dir


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_dir: Path,
    accelerator,
) -> dict:
    """Restore model, optimizer, scheduler, and RNG state in-place.

    Args:
        model: :class:`~formosa_dual.models.dual_model.DualObjectiveModel`.
        optimizer: PyTorch optimizer.
        scheduler: LR scheduler (or None).
        checkpoint_dir: Path to a ``checkpoint-*`` directory.
        accelerator: ``accelerate.Accelerator`` instance.

    Returns:
        ``training_state`` dict with keys: ``step``, ``epoch``, ``best_metric``.
    """
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # PEFT adapter
    backbone = model.backbone
    if hasattr(backbone, "load_adapter"):
        backbone.load_adapter(str(checkpoint_dir), adapter_name="default")
        logger.info("Loaded PEFT adapter from %s", checkpoint_dir)
    elif hasattr(backbone, "from_pretrained"):
        pass  # handled externally when needed

    # Aux modules
    aux_safetensors = checkpoint_dir / "aux_modules.safetensors"
    aux_pt = checkpoint_dir / "aux_modules.pt"
    if aux_safetensors.exists():
        from safetensors.torch import load_file as st_load
        flat = st_load(str(aux_safetensors))
        _load_aux_modules(model, flat, is_flat=True)
    elif aux_pt.exists():
        aux_state = torch.load(aux_pt, map_location="cpu")
        _load_aux_modules(model, aux_state, is_flat=False)

    # Optimizer + scheduler
    opt_path = checkpoint_dir / "optimizer.pt"
    if opt_path.exists():
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
    sched_path = checkpoint_dir / "scheduler.pt"
    if scheduler is not None and sched_path.exists():
        scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))

    # RNG state
    rng_path = checkpoint_dir / "rng_state.pt"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu")
        torch.set_rng_state(rng_state["cpu"])
        if "cuda" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["cuda"])

    # Training state
    ts_path = checkpoint_dir / "training_state.json"
    if ts_path.exists():
        training_state = json.loads(ts_path.read_text(encoding="utf-8"))
    else:
        training_state = {"step": 0, "epoch": 0, "best_metric": None}

    logger.info("Checkpoint loaded from %s (step=%s)", checkpoint_dir, training_state.get("step"))
    return training_state


def _load_aux_modules(model, state, is_flat: bool) -> None:
    """Load aux module state dicts into model in-place."""
    if is_flat:
        # Reconstruct per-submodule state dicts from flat keys
        pooler_sd: dict = {}
        proj_sd: dict = {}
        tag_sd: dict = {}
        for k, v in state.items():
            if k.startswith("pooler."):
                pooler_sd[k[len("pooler."):]] = v
            elif k.startswith("proj_head."):
                proj_sd[k[len("proj_head."):]] = v
            elif k.startswith("tag_projector_projector."):
                tag_sd[k[len("tag_projector_projector."):]] = v
    else:
        pooler_sd = state.get("pooler", {})
        proj_sd = state.get("proj_head", {})
        tag_sd = state.get("tag_projector_projector", {})

    if model.pooler is not None and pooler_sd:
        model.pooler.load_state_dict(pooler_sd, strict=True)
    if model.proj_head is not None and proj_sd:
        model.proj_head.load_state_dict(proj_sd, strict=True)
    if model.tag_projector is not None and tag_sd:
        model.tag_projector.projector.load_state_dict(tag_sd, strict=True)
