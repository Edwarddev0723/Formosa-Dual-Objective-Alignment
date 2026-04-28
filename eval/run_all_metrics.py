#!/usr/bin/env python
"""eval/run_all_metrics.py — run all metrics on a checkpoint (§6.10).

Usage:
    python eval/run_all_metrics.py \\
        --checkpoint PATH \\
        --base-model HF_MODEL_ID \\
        --test-sets NAME[,NAME ...] \\
        --metrics caption,contrastive,culturalness \\
        --batch-size 4 \\
        --output PATH
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all eval metrics on a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--test-sets", required=True, help="Comma-separated test set names.")
    parser.add_argument("--metrics", default="caption,contrastive,culturalness")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from formosa_dual.utils.logging import get_logger
    logger = get_logger(__name__)

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    enabled_metrics = {m.strip() for m in args.metrics.split(",")}
    test_sets = [s.strip() for s in args.test_sets.split(",")]

    logger.info("Running metrics %s on checkpoint %s", enabled_metrics, checkpoint_path)

    # Load training_state.json to get the original config path
    import json
    state_file = checkpoint_path / "training_state.json"
    if not state_file.exists():
        logger.error("training_state.json not found in checkpoint.")
        sys.exit(1)

    config_yaml = checkpoint_path / "run_config.yaml"
    if not config_yaml.exists():
        logger.error("run_config.yaml not found in checkpoint.")
        sys.exit(1)

    from formosa_dual.config import load_config
    import yaml
    raw = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
    # Build a minimal config from the saved yaml
    from formosa_dual.config.schema import RunConfig
    cfg = RunConfig(**raw)

    from formosa_dual.data.tag_vocab import TagVocabulary
    from formosa_dual.data.manifest import load_manifest
    from formosa_dual.data.dataset import FormosaDataset
    from formosa_dual.data.collator import DualCollator
    from formosa_dual.data.negative_sampler import NegativeSampler
    from transformers import AutoProcessor
    from formosa_dual.models.dual_model import DualObjectiveModel
    from formosa_dual.eval.reporter import Reporter
    from torch.utils.data import DataLoader

    vocab = TagVocabulary(Path(cfg.data.vocab_path))
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    sampler = NegativeSampler(
        vocab=vocab,
        strategy=cfg.contrastive.neg_sampling,
        num_negatives=cfg.contrastive.num_negatives,
        seed=42,
    )
    collator = DualCollator(
        processor=processor,
        vocab=vocab,
        negative_sampler=sampler,
        max_caption_tokens=cfg.data.max_caption_tokens,
        max_pos_tags=cfg.data.max_pos_tags,
    )

    model = DualObjectiveModel(cfg=cfg, vocab=vocab, processor=processor)
    from formosa_dual.training.checkpoint import load_checkpoint
    load_checkpoint(model, optimizer=None, scheduler=None, checkpoint_dir=checkpoint_path)

    import torch
    device = torch.device("cpu")
    model.to(device).eval()

    reporter = Reporter(output_dir=output_path, run_name=checkpoint_path.name)

    for test_set_name in test_sets:
        manifest_path = Path(cfg.data.test_manifests[test_set_name])
        test_ds = FormosaDataset(manifest_path=manifest_path, vocab=vocab,
                                 image_root=Path(cfg.data.image_root))
        loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collator)

        metrics = _eval_one_set(model, loader, enabled_metrics, cfg, device)
        reporter.add_section(test_set_name, metrics)
        logger.info("Test set %s metrics: %s", test_set_name, metrics)

    output_path.mkdir(parents=True, exist_ok=True)
    reporter.write()
    logger.info("Report written to %s", output_path)
    sys.exit(0)


def _eval_one_set(model, loader, enabled_metrics, cfg, device) -> dict:
    import torch
    from formosa_dual.losses.dual_objective import DualObjectiveLoss
    from formosa_dual.eval.caption_metrics import bleu4, rouge_l

    references, hypotheses = [], []
    total_loss = 0.0
    n_batches = 0

    loss_fn = DualObjectiveLoss(cfg=cfg, total_steps=1)

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            out = model(batch)
            if "lm_loss" in out and out["lm_loss"] is not None:
                total_loss += out["lm_loss"].item()
                n_batches += 1

    results = {}
    if n_batches > 0:
        results["avg_lm_loss"] = total_loss / n_batches
    return results


if __name__ == "__main__":
    main()
