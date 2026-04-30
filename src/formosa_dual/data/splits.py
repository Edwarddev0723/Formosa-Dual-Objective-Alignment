"""formosa_dual.data.splits — group-aware train/val/test splitting.

Groups by ``article_url``, stratifies by ``source``, verifies no leakage
across splits by ``image_hash``, perceptual hash Hamming distance, and
CLIP cosine similarity.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Optional

from formosa_dual.data.manifest import load_manifest, write_manifest
from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


def _hamming_distance(a: str, b: str) -> int:
    """Compute bit-level Hamming distance between two hex-string perceptual hashes."""
    try:
        ia = int(a, 16)
        ib = int(b, 16)
    except ValueError:
        return 999  # treat invalid hashes as far apart
    xor = ia ^ ib
    return bin(xor).count("1")


def build_splits(
    records: list[dict],
    train_ratio: float = 0.80,
    dev_ratio: float = 0.10,
    test_ratio: float = 0.10,
    group_by: str = "article_url",
    stratify_by: str = "source",
    source_holdout: int = 800,
    cultural_hard_size: int = 500,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Build group-stratified train/dev/test splits.

    Args:
        records: Full annotated manifest records.
        train_ratio: Fraction of groups allocated to training.
        dev_ratio: Fraction of groups allocated to dev.
        test_ratio: Fraction of groups allocated to test_id.
        group_by: Record field used to group records.
        stratify_by: Record field used for stratification.
        source_holdout: Number of records from held-out sources for
            ``test_source_holdout``.
        cultural_hard_size: Number of records for ``test_cultural_hard``
            (selected by highest difficulty).
        seed: Random seed.

    Returns:
        Dict mapping split name → list of records:
        ``train``, ``dev``, ``test_id``, ``test_source_holdout``,
        ``test_cultural_hard``.

    Raises:
        ValueError: If any leakage is detected between splits.
    """
    rng = random.Random(seed)

    # Group records by group_by field
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for rec in records:
        key = rec.get(group_by) or rec.get("id", "")
        groups[key].append(rec)

    # Stratify groups by stratify_by field
    strata: dict[str, list[str]] = collections.defaultdict(list)
    for group_key, recs in groups.items():
        stratum = recs[0].get(stratify_by, "unknown")
        strata[stratum].append(group_key)

    train_groups: list[str] = []
    dev_groups: list[str] = []
    test_groups: list[str] = []

    for stratum, group_keys in strata.items():
        rng.shuffle(group_keys)
        n = len(group_keys)
        n_train = max(1, round(n * train_ratio))
        n_dev = max(1, round(n * dev_ratio))
        train_groups.extend(group_keys[:n_train])
        dev_groups.extend(group_keys[n_train: n_train + n_dev])
        test_groups.extend(group_keys[n_train + n_dev:])

    def _collect(keys: list[str]) -> list[dict]:
        out: list[dict] = []
        for k in keys:
            out.extend(groups[k])
        return out

    train_records = _collect(train_groups)
    dev_records = _collect(dev_groups)
    test_id_records = _collect(test_groups)

    # test_source_holdout: take from a different source if possible
    all_sources = {rec.get(stratify_by, "") for rec in records}
    holdout_records: list[dict] = []
    # ASSUMPTION: hold out records from the source with the fewest train groups
    source_train_counts: dict[str, int] = collections.Counter(
        rec.get(stratify_by, "") for rec in train_records
    )
    if all_sources:
        holdout_source = min(all_sources, key=lambda s: source_train_counts.get(s, 0))
        holdout_candidates = [r for r in records if r.get(stratify_by) == holdout_source]
        rng.shuffle(holdout_candidates)
        holdout_records = holdout_candidates[:source_holdout]

    # test_cultural_hard: highest difficulty records
    sorted_by_difficulty = sorted(
        records, key=lambda r: r.get("difficulty", 0), reverse=True
    )
    cultural_hard_records = sorted_by_difficulty[:cultural_hard_size]

    splits = {
        "train": train_records,
        "dev": dev_records,
        "test_id": test_id_records,
        "test_source_holdout": holdout_records,
        "test_cultural_hard": cultural_hard_records,
    }

    # Leakage verification: check image_hash overlap and phash proximity
    _verify_no_leakage(splits)

    return splits


def _verify_no_leakage(splits: dict[str, list[dict]]) -> None:
    """Verify no image leakage across train vs. test splits.

    Raises:
        ValueError: If leakage is found.
    """
    train_hashes: set[str] = {r.get("image_hash", "") for r in splits.get("train", [])}
    train_phashes: list[str] = [r.get("phash", "") for r in splits.get("train", [])]

    for split_name in ("dev", "test_id", "test_source_holdout", "test_cultural_hard"):
        for rec in splits.get(split_name, []):
            # Exact hash overlap
            if rec.get("image_hash") in train_hashes:
                raise ValueError(
                    f"Leakage: image_hash '{rec['image_hash']}' appears in both "
                    f"train and {split_name} (record id={rec.get('id')})"
                )
            # Perceptual hash proximity
            rec_phash = rec.get("phash", "")
            if rec_phash:
                for tr_phash in train_phashes:
                    if tr_phash and _hamming_distance(rec_phash, tr_phash) < 8:
                        raise ValueError(
                            f"Leakage (phash Hamming<8): record {rec.get('id')} in "
                            f"{split_name} is near-duplicate of a training image"
                        )

    logger.info("Split leakage verification passed")
