#!/usr/bin/env python
"""eval/retrieval_only.py — retrieval-only evaluation (spec §6.11).

Usage:
    python eval/retrieval_only.py \\
        --checkpoint PATH \\
        --base-model HF_MODEL_ID \\
        --test-set NAME \\
        --output PATH

Computes only retrieval metrics (faster than full eval; useful for ablations).

Exit codes:
    0 — success
    1 — user / config error (missing files, bad args)
    2 — internal error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrieval-only evaluation on a checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", required=True, help="Path to a checkpoint-* directory."
    )
    p.add_argument("--base-model", required=True, help="HuggingFace base model id.")
    p.add_argument(
        "--test-set",
        required=True,
        help="Configured test-set name or manifest JSONL path.",
    )
    p.add_argument("--output", required=True, help="Output JSON path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        from formosa_dual.utils.logging import get_logger
    except ImportError as exc:
        print(f"Critical import error: {exc}", file=sys.stderr)
        sys.exit(2)

    logger = get_logger("eval.retrieval_only")

    ckpt_dir = Path(args.checkpoint)
    if not ckpt_dir.is_dir():
        logger.error("Checkpoint dir not found: %s", ckpt_dir.resolve())
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import yaml
        from torch.utils.data import DataLoader
        from transformers import AutoProcessor

        from formosa_dual.config.schema import RunConfig
        from formosa_dual.data.collator import DualCollator
        from formosa_dual.data.dataset import FormosaDataset
        from formosa_dual.data.negative_sampler import NegativeSampler
        from formosa_dual.data.tag_vocab import TagVocabulary
        from formosa_dual.eval.retrieval_metrics import evaluate_retrieval_loader
        from formosa_dual.models.dual_model import DualObjectiveModel
        from formosa_dual.training.checkpoint import load_checkpoint
    except ImportError as exc:
        logger.error("formosa_dual import failed: %s", exc)
        sys.exit(2)

    try:
        cfg = _load_checkpoint_config(ckpt_dir, args.base_model, yaml, RunConfig)
        vocab = TagVocabulary(Path(cfg.data.vocab_path))
        manifest_path = _resolve_test_manifest(args.test_set, cfg)
        if not manifest_path.exists():
            logger.error("Test manifest not found: %s", manifest_path.resolve())
            sys.exit(1)

        device = _select_eval_device(torch, cfg)
        logger.info("Retrieval eval device: %s", device)

        processor = AutoProcessor.from_pretrained(
            args.base_model, trust_remote_code=True
        )
        sampler = NegativeSampler(
            vocab=vocab,
            strategy="uniform",
            num_negatives=1,
            seed=cfg.training.seed,
        )
        collator = DualCollator(
            processor=processor,
            vocab=vocab,
            negative_sampler=sampler,
            max_caption_tokens=cfg.caption.max_caption_tokens,
            max_pos_tags=10,
        )
        dataset = FormosaDataset(
            manifest_path=manifest_path,
            vocab=vocab,
            image_root=Path(cfg.data.image_root),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.training.per_device_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )

        model = DualObjectiveModel(cfg=cfg, vocab=vocab, processor=processor)
        model.to(device)
        load_checkpoint(
            model=model,
            optimizer=None,
            scheduler=None,
            checkpoint_dir=ckpt_dir,
            accelerator=None,
        )
        model.eval()

        metrics = evaluate_retrieval_loader(
            model=model,
            dataloader=dataloader,
            vocab=vocab,
            device=device,
            desc=f"retrieval:{manifest_path.stem}",
            show_progress=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Retrieval evaluation failed: %s", exc, exc_info=True)
        sys.exit(2)

    logger.info("Loaded %d records from %s", len(dataset), manifest_path)
    logger.info("Checkpoint dir: %s", ckpt_dir)
    logger.info("Base model: %s", args.base_model)

    report = {
        "checkpoint": str(ckpt_dir),
        "base_model": args.base_model,
        "test_set": str(manifest_path),
        "metrics": metrics,
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Wrote retrieval-only report: %s", output_path)
    sys.exit(0)


def _load_checkpoint_config(ckpt_dir: Path, base_model: str, yaml, run_config_cls):
    config_path = ckpt_dir / "run_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"run_config.yaml not found in checkpoint: {ckpt_dir}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw.setdefault("model", {})["name"] = base_model
    return run_config_cls.model_validate(raw)


def _resolve_test_manifest(test_set: str, cfg) -> Path:
    candidate = Path(test_set)
    if candidate.exists():
        return candidate
    manifest_map = getattr(cfg.data, "test_manifests", {}) or {}
    if test_set in manifest_map:
        return Path(manifest_map[test_set])
    return candidate


def _select_eval_device(torch_module, cfg):
    forced = getattr(cfg.device, "force", None)
    if forced == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("Config forces CUDA, but CUDA is not available")
    if forced == "mps" and not torch_module.backends.mps.is_available():
        raise RuntimeError("Config forces MPS, but MPS is not available")
    if forced is not None:
        return torch_module.device(forced)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


if __name__ == "__main__":
    main()
