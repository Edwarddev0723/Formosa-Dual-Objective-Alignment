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
        help="Configured test-set name, comma-separated names, 'all', or manifest JSONL path.",
    )
    p.add_argument("--output", required=True, help="Output JSON path.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override eval batch size. Defaults to training.per_device_batch_size.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override DataLoader workers. Defaults to data.num_workers.",
    )
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
        from formosa_dual.eval.retrieval_metrics import (
            embed_all_tag_embeddings,
            evaluate_retrieval_loader,
        )
        from formosa_dual.models.dual_model import DualObjectiveModel
        from formosa_dual.training.checkpoint import load_checkpoint
    except ImportError as exc:
        logger.error("formosa_dual import failed: %s", exc)
        sys.exit(2)

    try:
        cfg = _load_checkpoint_config(ckpt_dir, args.base_model, yaml, RunConfig)
        vocab = TagVocabulary(Path(cfg.data.vocab_path))
        test_specs = _resolve_test_manifests(args.test_set, cfg)
        for _, manifest_path in test_specs:
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

        tag_gallery = embed_all_tag_embeddings(model, vocab, device=device)
        if tag_gallery is None:
            raise RuntimeError("Checkpoint does not expose tag_projector embeddings")

        split_reports = {}
        for test_name, manifest_path in test_specs:
            dataset = FormosaDataset(
                manifest_path=manifest_path,
                vocab=vocab,
                image_root=Path(cfg.data.image_root),
            )
            dataloader = _build_dataloader(
                DataLoader,
                dataset,
                collator,
                cfg,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            metrics = evaluate_retrieval_loader(
                model=model,
                dataloader=dataloader,
                vocab=vocab,
                device=device,
                desc=f"retrieval:{test_name}",
                show_progress=True,
                tag_gallery=tag_gallery,
            )
            split_reports[test_name] = {
                "manifest": str(manifest_path),
                "n_records": len(dataset),
                "metrics": metrics,
            }
            logger.info("Loaded %d records from %s", len(dataset), manifest_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Retrieval evaluation failed: %s", exc, exc_info=True)
        sys.exit(2)

    logger.info("Checkpoint dir: %s", ckpt_dir)
    logger.info("Base model: %s", args.base_model)

    report = {
        "checkpoint": str(ckpt_dir),
        "base_model": args.base_model,
        "test_set": args.test_set,
        "splits": split_reports,
    }
    if len(split_reports) == 1:
        only = next(iter(split_reports.values()))
        report["metrics"] = only["metrics"]
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


def _resolve_test_manifests(test_set: str, cfg) -> list[tuple[str, Path]]:
    manifest_map = getattr(cfg.data, "test_manifests", {}) or {}
    if test_set == "all":
        if not manifest_map:
            raise ValueError(
                "--test-set all requested, but config has no test_manifests"
            )
        return [(name, Path(path)) for name, path in manifest_map.items()]

    specs: list[tuple[str, Path]] = []
    for raw_name in test_set.split(","):
        name = raw_name.strip()
        if not name:
            continue
        manifest_path = _resolve_test_manifest(name, cfg)
        report_name = name if name in manifest_map else manifest_path.stem
        specs.append((report_name, manifest_path))
    if not specs:
        raise ValueError("--test-set resolved to no manifests")
    return specs


def _resolve_test_manifest(test_set: str, cfg) -> Path:
    candidate = Path(test_set)
    if candidate.exists():
        return candidate
    manifest_map = getattr(cfg.data, "test_manifests", {}) or {}
    if test_set in manifest_map:
        return Path(manifest_map[test_set])
    return candidate


def _build_dataloader(
    data_loader_cls,
    dataset,
    collator,
    cfg,
    batch_size: int | None = None,
    num_workers: int | None = None,
):
    effective_batch_size = batch_size or cfg.training.per_device_batch_size
    effective_num_workers = (
        cfg.data.num_workers if num_workers is None else max(int(num_workers), 0)
    )
    kwargs = {
        "batch_size": effective_batch_size,
        "shuffle": False,
        "collate_fn": collator,
        "num_workers": effective_num_workers,
        "pin_memory": cfg.data.pin_memory,
    }
    if effective_num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return data_loader_cls(dataset, **kwargs)


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
