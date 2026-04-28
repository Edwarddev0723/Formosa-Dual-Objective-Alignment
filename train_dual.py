#!/usr/bin/env python
"""train_dual.py — main training entrypoint for formosa-dual (§6.8).

Usage:
    python train_dual.py \\
        --profile {dev_mac,dev_smoke,prod_gb10} \\
        --experiment NAME \\
        [--override KEY=VALUE [KEY=VALUE ...]] \\
        [--smoke] \\
        [--resume-from PATH] \\
        [--dry-run]

Exit codes:
  0 — training and final eval succeeded
  1 — config error or environment error
  2 — training failed mid-way
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train the formosa-dual model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--profile", required=True, choices=["dev_mac", "dev_smoke", "prod_gb10"])
    parser.add_argument("--experiment", required=True, help="Experiment config name (e.g. v3_hero).")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config keys (e.g. training.num_epochs=5).")
    parser.add_argument("--smoke", action="store_true", help="Force smoke mode.")
    parser.add_argument("--resume-from", default=None, help="Path to a checkpoint directory.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load + print resolved config then exit without training.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # --- Step 1-2: Load config ---
    try:
        from formosa_dual.config import ConfigError, load_config
    except ImportError as exc:
        print(f"Critical import error: {exc}", file=sys.stderr)
        sys.exit(2)

    try:
        cfg = load_config(
            profile=args.profile,
            experiment=args.experiment,
            cli_overrides=args.override,
            smoke=args.smoke,
            allow_no_loss=args.dry_run,  # only bypass at_least_one_loss in --dry-run
        )
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(1)

    from formosa_dual.utils.logging import get_logger
    logger = get_logger(__name__)

    # --- Step 3: Device capability report ---
    from formosa_dual.training.device import device_capability_report
    report = device_capability_report()
    for k, v in report.items():
        logger.info("env | %s = %s", k, v)

    # --- Step 4: Validate config for device ---
    from formosa_dual.config.validation import validate_config_for_device
    from formosa_dual.training.device import select_device
    device = select_device(cfg.device)
    try:
        validate_config_for_device(cfg, device)
    except ConfigError as exc:
        logger.error("Device config error: %s", exc)
        sys.exit(1)

    # --- Step 5: Set seed ---
    from formosa_dual.utils.seeding import set_seed
    set_seed(cfg.training.seed)

    # --- Step 6: Dry run ---
    if args.dry_run:
        import yaml
        # Hard-stop guard: refuse to actually train when both losses are disabled
        # (we got here only via allow_no_loss=True).
        if not cfg.caption.enabled and not cfg.contrastive.enabled:
            logger.warning(
                "Both caption and contrastive losses are disabled — dry-run only. "
                "This config cannot be used for actual training."
            )
        print(yaml.safe_dump(cfg.model_dump(), allow_unicode=True))
        sys.exit(0)

    # Hard-stop guard for actual training: never allow no-loss training to start
    if not cfg.caption.enabled and not cfg.contrastive.enabled:
        logger.error(
            "Cannot train with both caption and contrastive disabled. "
            "Use --dry-run to validate this config without training."
        )
        sys.exit(1)

    # --- Step 7: Build accelerator ---
    from formosa_dual.training.accelerator import build_accelerator
    accelerator = build_accelerator(cfg)
    # accelerator.device should match `device` selected above; trust accelerator's value.
    device = accelerator.device

    # --- Step 8: Build vocab, dataset, dataloader, collator ---
    from formosa_dual.data.tag_vocab import TagVocabulary
    from formosa_dual.data.manifest import load_manifest
    from formosa_dual.data.dataset import FormosaDataset
    from formosa_dual.data.negative_sampler import NegativeSampler
    from formosa_dual.data.collator import DualCollator
    from torch.utils.data import DataLoader

    vocab = TagVocabulary(Path(cfg.data.vocab_path))

    # Smoke caps
    smoke_max_train = cfg.smoke.max_train_samples if cfg.smoke.enabled else None
    smoke_max_eval = cfg.smoke.max_eval_samples if cfg.smoke.enabled else None

    train_ds = FormosaDataset(
        manifest_path=Path(cfg.data.train_manifest),
        vocab=vocab,
        image_root=Path(cfg.data.image_root),
        smoke_max_samples=smoke_max_train,
    )
    val_ds = FormosaDataset(
        manifest_path=Path(cfg.data.val_manifest),
        vocab=vocab,
        image_root=Path(cfg.data.image_root),
        smoke_max_samples=smoke_max_eval,
    )

    sampler = NegativeSampler(
        vocab=vocab,
        strategy=cfg.contrastive.neg_sampling,
        num_negatives=cfg.contrastive.negatives_per_image,
        seed=cfg.training.seed,
    )

    # Effective max caption tokens (smoke override may shrink it)
    max_caption_tokens = cfg.caption.max_caption_tokens
    if cfg.smoke.enabled and cfg.smoke.max_caption_tokens_override is not None:
        max_caption_tokens = cfg.smoke.max_caption_tokens_override

    # Processor is loaded both here (for the collator) and inside
    # DualObjectiveModel.load_backbone(); transformers caches so this is cheap.
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(cfg.model.name, trust_remote_code=True)

    # ASSUMPTION: not specified in spec §4.1; using DualCollator default of 10
    # (spec §5.9). Extend AuxModulesConfig if a per-experiment override becomes needed.
    collator = DualCollator(
        processor=processor,
        vocab=vocab,
        negative_sampler=sampler,
        max_caption_tokens=max_caption_tokens,
        max_pos_tags=10,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.per_device_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    # --- Step 9: Build model ---
    from formosa_dual.models.dual_model import DualObjectiveModel
    model = DualObjectiveModel(cfg=cfg, vocab=vocab, processor=processor)
    model.to(device)

    # --- Step 10: Build loss ---
    from formosa_dual.losses.dual_objective import DualObjectiveLoss
    import math
    total_steps = math.ceil(len(train_loader) / cfg.training.gradient_accumulation_steps) * cfg.training.num_epochs
    if cfg.smoke.enabled:
        total_steps = cfg.smoke.max_steps
    loss_fn = DualObjectiveLoss(cfg=cfg, total_steps=total_steps)

    # --- Step 13 (check resume before building trainer) ---
    resume_path = Path(args.resume_from) if args.resume_from else None

    # --- Step 14: Build trainer and train ---
    from formosa_dual.training.trainer import DualTrainer
    trainer = DualTrainer(
        cfg=cfg,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        accelerator=accelerator,
        vocab=vocab,
    )

    if resume_path is not None:
        logger.info("Resuming from checkpoint: %s", resume_path)
        trainer.load_checkpoint(resume_path)

    try:
        trainer.train()
    except SystemExit:
        # NaN exit (code 2) propagated from trainer
        raise
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        sys.exit(2)

    # --- Step 15: Final eval ---
    if cfg.smoke.enabled and cfg.smoke.skip_eval:
        logger.info("Smoke mode with skip_eval=True; skipping test-set evaluation.")
        test_manifests: dict = {}
    else:
        test_manifests = getattr(cfg.data, "test_manifests", {}) or {}
    for test_name, manifest_path in test_manifests.items():
        if not Path(manifest_path).exists():
            logger.warning(
                "Test manifest %s not found at %s; skipping.", test_name, manifest_path
            )
            continue
        test_ds = FormosaDataset(
            manifest_path=Path(manifest_path),
            vocab=vocab,
            image_root=Path(cfg.data.image_root),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.training.per_device_batch_size,
            collate_fn=collator,
            num_workers=cfg.data.num_workers,
        )
        trainer.val_loader = accelerator.prepare(test_loader)
        metrics = trainer.evaluate(test_name)
        logger.info("Test set %s: %s", test_name, metrics)

    # --- Step 16: Write report ---
    from formosa_dual.eval.reporter import Reporter
    output_dir = Path(cfg.logging.output_dir) / (cfg.logging.run_name or "run")
    reporter = Reporter(output_dir=output_dir, run_name=cfg.logging.run_name or "run")
    reporter.add_section("final_val", trainer._last_val_metrics)
    reporter.write()
    logger.info("Training complete. Reports in %s", output_dir)
    sys.exit(0)


if __name__ == "__main__":
    main()
