#!/usr/bin/env python3
"""Generate synthetic training/validation data for smoke tests and CI.

Usage::

    python scripts/make_synthetic_data.py \\
        --num-train 8 --num-val 4 \\
        --output-dir data/synthetic

Produces:
    PNG images (224x224, deterministic from index seed),
    template captions,
    synthetic vocab with 16 tags,
    manifests train_synth.jsonl, val_synth.jsonl,
    vocab vocab_synth.json.

Must run in <10 seconds on Mac.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from formosa_dual.data.manifest import write_manifest
from formosa_dual.data.tag_vocab import TagVocabulary
from formosa_dual.utils.logging import get_logger
from formosa_dual.utils.synthetic import make_synthetic_record

logger = get_logger("make_synthetic_data")

# 16 synthetic tags (two per category for diversity)
_SYNTHETIC_TAGS = [
    {"id": 0,  "tag": "媽祖",   "tier": 1, "freq": 50, "category": "宗教"},
    {"id": 1,  "tag": "城隍",   "tier": 1, "freq": 40, "category": "宗教"},
    {"id": 2,  "tag": "廟宇",   "tier": 1, "freq": 60, "category": "建築"},
    {"id": 3,  "tag": "古蹟",   "tier": 1, "freq": 45, "category": "建築"},
    {"id": 4,  "tag": "台北",   "tier": 1, "freq": 80, "category": "地理"},
    {"id": 5,  "tag": "台南",   "tier": 1, "freq": 70, "category": "地理"},
    {"id": 6,  "tag": "傳統市場", "tier": 2, "freq": 30, "category": "飲食"},
    {"id": 7,  "tag": "夜市",   "tier": 2, "freq": 35, "category": "飲食"},
    {"id": 8,  "tag": "清代",   "tier": 2, "freq": 25, "category": "歷史"},
    {"id": 9,  "tag": "日治",   "tier": 2, "freq": 20, "category": "歷史"},
    {"id": 10, "tag": "原住民", "tier": 2, "freq": 15, "category": "族群"},
    {"id": 11, "tag": "閩南",   "tier": 2, "freq": 18, "category": "族群"},
    {"id": 12, "tag": "龍山寺", "tier": 3, "freq": 12, "category": "景點"},
    {"id": 13, "tag": "九份",   "tier": 3, "freq": 10, "category": "景點"},
    {"id": 14, "tag": "歌仔戲", "tier": 3, "freq": 8,  "category": "表演"},
    {"id": 15, "tag": "布袋戲", "tier": 3, "freq": 7,  "category": "表演"},
]

_VOCAB_TAGS = [e["tag"] for e in _SYNTHETIC_TAGS]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for smoke tests."
    )
    parser.add_argument("--num-train", type=int, default=8, help="Number of training samples")
    parser.add_argument("--num-val", type=int, default=4, help="Number of validation samples")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/synthetic"),
        help="Directory to write outputs"
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"

    # 1. Write vocab
    vocab_data = {"version": "v1", "size": len(_SYNTHETIC_TAGS), "tags": _SYNTHETIC_TAGS}
    vocab_path = out_dir / "vocab_synth.json"
    with vocab_path.open("w", encoding="utf-8") as fh:
        json.dump(vocab_data, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote vocab to %s", vocab_path)

    # 2. Generate train records
    train_records = [
        make_synthetic_record(i, _VOCAB_TAGS, image_dir=img_dir, tags_per_image=3)
        for i in range(args.num_train)
    ]
    write_manifest(train_records, out_dir / "train_synth.jsonl")

    # 3. Generate val records (offset index to avoid hash collisions with train)
    val_records = [
        make_synthetic_record(i + args.num_train, _VOCAB_TAGS, image_dir=img_dir, tags_per_image=3)
        for i in range(args.num_val)
    ]
    write_manifest(val_records, out_dir / "val_synth.jsonl")

    logger.info(
        "Synthetic data written to %s (%d train, %d val, %d tags)",
        out_dir, args.num_train, args.num_val, len(_SYNTHETIC_TAGS)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
