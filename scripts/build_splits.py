#!/usr/bin/env python
"""scripts/build_splits.py — group-stratified train/dev/test splits (§6.5).

Usage:
    python scripts/build_splits.py \\
        --annotations PATH \\
        --train-ratio 0.80 \\
        --dev-ratio 0.10 \\
        --test-ratio 0.10 \\
        --group-by article_url \\
        --stratify-by source \\
        --source-holdout 800 \\
        --cultural-hard-size 500 \\
        --seed 42 \\
        --output-dir PATH
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build group-stratified dataset splits.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--group-by", default="article_url")
    parser.add_argument("--stratify-by", default="source")
    parser.add_argument("--source-holdout", type=int, default=800)
    parser.add_argument("--cultural-hard-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    from formosa_dual.data.manifest import load_manifest, write_manifest
    from formosa_dual.data.splits import build_splits

    annotations_path = Path(args.annotations)
    output_dir = Path(args.output_dir)

    if not annotations_path.exists():
        logger.error("Annotations not found: %s", annotations_path)
        sys.exit(1)

    records = load_manifest(annotations_path)
    logger.info("Loaded %d records", len(records))

    splits = build_splits(
        records=records,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        group_by=args.group_by,
        stratify_by=args.stratify_by,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, split_records in splits.items():
        out_path = output_dir / f"{name}.jsonl"
        write_manifest(split_records, out_path)
        logger.info("Wrote %d records to %s", len(split_records), out_path)

    # Manifest JSON
    manifest_info = {
        "seed": args.seed,
        "sizes": {k: len(v) for k, v in splits.items()},
    }
    manifest_path = output_dir / "splits_manifest.json"
    manifest_path.write_text(json.dumps(manifest_info, indent=2), encoding="utf-8")
    logger.info("Splits manifest: %s", manifest_path)
    sys.exit(0)


if __name__ == "__main__":
    main()
