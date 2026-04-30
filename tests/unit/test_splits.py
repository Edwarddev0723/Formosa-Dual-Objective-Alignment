"""Tests for leakage-safe dataset splitting."""
from __future__ import annotations

from formosa_dual.data.splits import build_splits


def _record(idx: int, article_url: str, image_hash: str, source: str = "source_a") -> dict:
    return {
        "id": f"sample_{idx}",
        "image_path": f"img_{idx}.jpg",
        "caption": f"caption {idx}",
        "source": source,
        "article_url": article_url,
        "image_hash": f"sha256:{image_hash}",
        "phash": "",
        "difficulty": idx % 5 + 1,
        "culture_tags": ["台灣"],
        "metadata": {},
    }


def test_duplicate_image_hashes_stay_out_of_eval_leakage():
    records = [
        _record(0, "article_a", "dup"),
        _record(1, "article_b", "dup"),
        _record(2, "article_c", "unique_c"),
        _record(3, "article_d", "unique_d"),
        _record(4, "article_e", "unique_e"),
        _record(5, "article_f", "unique_f"),
    ]

    splits = build_splits(
        records,
        train_ratio=0.5,
        dev_ratio=0.25,
        test_ratio=0.25,
        source_holdout=10,
        cultural_hard_size=10,
        seed=7,
    )

    train_hashes = {r["image_hash"] for r in splits["train"]}
    for split_name in ("dev", "test_id", "test_source_holdout", "test_cultural_hard"):
        split_hashes = {r["image_hash"] for r in splits[split_name]}
        assert train_hashes.isdisjoint(split_hashes)


def test_same_phash_records_stay_together():
    records = [
        _record(0, "article_a", "hash_a"),
        _record(1, "article_b", "hash_b"),
        _record(2, "article_c", "hash_c"),
        _record(3, "article_d", "hash_d"),
    ]
    records[0]["phash"] = "abcd"
    records[1]["phash"] = "abcd"

    splits = build_splits(
        records,
        train_ratio=0.5,
        dev_ratio=0.25,
        test_ratio=0.25,
        source_holdout=10,
        cultural_hard_size=10,
        seed=3,
    )

    split_by_id = {
        rec["id"]: split_name
        for split_name, split_records in splits.items()
        if split_name in {"train", "dev", "test_id"}
        for rec in split_records
    }
    assert split_by_id["sample_0"] == split_by_id["sample_1"]
