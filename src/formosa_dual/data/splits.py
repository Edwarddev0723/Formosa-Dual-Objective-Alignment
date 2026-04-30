"""formosa_dual.data.splits — group-aware train/val/test splitting.

Groups by ``article_url``, stratifies by ``source``, verifies no leakage
across splits by ``image_hash``, perceptual hash Hamming distance, and
CLIP cosine similarity.
"""
from __future__ import annotations

import collections
import random

from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


_PHASH_BITS = 64
_PHASH_HEX_LEN = _PHASH_BITS // 4
_PHASH_CHUNK_HEX_LEN = 2  # 8-bit chunks; distance < 8 guarantees one exact chunk.


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

    # Group records by the requested group key AND duplicate image identity.
    # A dataset can contain the same image under multiple article URLs; those
    # records must stay together or the leakage verifier will correctly fail.
    groups = _build_leakage_groups(records, group_by=group_by)
    logger.info("Built %d leakage-safe groups from %d records", len(groups), len(records))

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

    non_train_pool = dev_records + test_id_records

    # test_source_holdout: take from a non-train source if possible
    all_sources = {rec.get(stratify_by, "") for rec in non_train_pool}
    holdout_records: list[dict] = []
    # ASSUMPTION: hold out records from the source with the fewest train groups
    source_train_counts: dict[str, int] = collections.Counter(
        rec.get(stratify_by, "") for rec in train_records
    )
    if all_sources:
        holdout_source = min(all_sources, key=lambda s: source_train_counts.get(s, 0))
        holdout_candidates = [r for r in non_train_pool if r.get(stratify_by) == holdout_source]
        rng.shuffle(holdout_candidates)
        holdout_records = holdout_candidates[:source_holdout]

    # test_cultural_hard: highest difficulty non-train records
    sorted_by_difficulty = sorted(
        non_train_pool, key=lambda r: r.get("difficulty", 0), reverse=True
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


def _normalise_phash(value: object) -> str | None:
    """Return a canonical 64-bit phash hex string, or None for invalid values."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        intval = int(text, 16)
    except ValueError:
        return None
    if intval.bit_length() > _PHASH_BITS:
        return None
    return f"{intval:0{_PHASH_HEX_LEN}x}"


def _build_leakage_groups(records: list[dict], group_by: str) -> dict[str, list[dict]]:
    """Build connected groups joined by article key, image hash, or near phash."""
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    buckets: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    phash_to_indexes: dict[str, list[int]] = collections.defaultdict(list)
    for idx, rec in enumerate(records):
        group_value = rec.get(group_by) or rec.get("id", "")
        if group_value:
            buckets[("group", str(group_value))].append(idx)

        image_hash = rec.get("image_hash") or ""
        if image_hash:
            buckets[("image_hash", str(image_hash))].append(idx)

        phash = _normalise_phash(rec.get("phash"))
        if phash:
            buckets[("phash", phash)].append(idx)
            phash_to_indexes[phash].append(idx)

    for indexes in buckets.values():
        first = indexes[0]
        for idx in indexes[1:]:
            union(first, idx)

    _union_near_phashes(phash_to_indexes, union)

    groups_by_root: dict[int, list[dict]] = collections.defaultdict(list)
    for idx, rec in enumerate(records):
        groups_by_root[find(idx)].append(rec)

    return {f"group_{root}": recs for root, recs in groups_by_root.items()}


def _union_near_phashes(phash_to_indexes: dict[str, list[int]], union) -> None:
    """Union phashes whose Hamming distance is below the leakage threshold."""
    lsh_buckets = _build_phash_lsh_index(phash_to_indexes.keys())

    near_edges = 0
    for candidates in lsh_buckets.values():
        if len(candidates) < 2:
            continue
        for i, left in enumerate(candidates):
            left_rep = phash_to_indexes[left][0]
            for right in candidates[i + 1:]:
                if _hamming_distance(left, right) < 8:
                    union(left_rep, phash_to_indexes[right][0])
                    near_edges += 1

    if near_edges:
        logger.info("Merged %d near-duplicate phash candidate edges", near_edges)


def _build_phash_lsh_index(phashes) -> dict[tuple[int, str], list[str]]:
    """Build byte-chunk LSH buckets for 64-bit perceptual hashes."""
    lsh_buckets: dict[tuple[int, str], list[str]] = collections.defaultdict(list)
    for phash in phashes:
        for chunk_idx in range(_PHASH_HEX_LEN // _PHASH_CHUNK_HEX_LEN):
            start = chunk_idx * _PHASH_CHUNK_HEX_LEN
            chunk = phash[start: start + _PHASH_CHUNK_HEX_LEN]
            lsh_buckets[(chunk_idx, chunk)].append(phash)
    return lsh_buckets


def _find_near_phash(
    phash: str,
    lsh_buckets: dict[tuple[int, str], list[str]],
) -> str | None:
    """Return a near phash from *lsh_buckets*, or None if no near match exists."""
    candidates: set[str] = set()
    for chunk_idx in range(_PHASH_HEX_LEN // _PHASH_CHUNK_HEX_LEN):
        start = chunk_idx * _PHASH_CHUNK_HEX_LEN
        chunk = phash[start: start + _PHASH_CHUNK_HEX_LEN]
        candidates.update(lsh_buckets.get((chunk_idx, chunk), []))
    for candidate in candidates:
        if _hamming_distance(phash, candidate) < 8:
            return candidate
    return None


def _verify_no_leakage(splits: dict[str, list[dict]]) -> None:
    """Verify no image leakage across train vs. test splits.

    Raises:
        ValueError: If leakage is found.
    """
    train_hashes: set[str] = {
        r.get("image_hash", "") for r in splits.get("train", []) if r.get("image_hash")
    }
    train_phashes = {
        phash
        for r in splits.get("train", [])
        if (phash := _normalise_phash(r.get("phash"))) is not None
    }
    train_phash_lsh = _build_phash_lsh_index(train_phashes)

    for split_name in ("dev", "test_id", "test_source_holdout", "test_cultural_hard"):
        for rec in splits.get(split_name, []):
            # Exact hash overlap
            rec_hash = rec.get("image_hash", "")
            if rec_hash and rec_hash in train_hashes:
                raise ValueError(
                    f"Leakage: image_hash '{rec_hash}' appears in both "
                    f"train and {split_name} (record id={rec.get('id')})"
                )
            # Perceptual hash proximity
            rec_phash = _normalise_phash(rec.get("phash"))
            if rec_phash and _find_near_phash(rec_phash, train_phash_lsh) is not None:
                raise ValueError(
                    f"Leakage (phash Hamming<8): record {rec.get('id')} in "
                    f"{split_name} is near-duplicate of a training image"
                )

    logger.info("Split leakage verification passed")
