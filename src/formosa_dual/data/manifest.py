"""formosa_dual.data.manifest — JSONL manifest I/O and validation.

Each line in a manifest file is a JSON object conforming to the schema
described in §5.4 of the construction spec.
"""
from __future__ import annotations

import json
from pathlib import Path

from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)

# Required top-level fields in every manifest record.
_REQUIRED_FIELDS = {
    "id",
    "image_path",
    "caption",
    "source",
    "image_hash",
    "difficulty",
    "culture_tags",
}


def load_manifest(path: Path) -> list[dict]:
    """Load a JSONL manifest from *path*.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        List of record dicts (one per line).

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If any line is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path.resolve()}")

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{lineno}: {exc}"
                ) from exc

    logger.info("Loaded %d records from %s", len(records), path)
    return records


def write_manifest(records: list[dict], path: Path) -> None:
    """Write *records* as JSONL to *path*.

    Parent directories are created if they do not exist.

    Args:
        records: List of record dicts.
        path: Destination ``.jsonl`` file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def validate_manifest(records: list[dict], vocab: "TagVocabulary | None" = None) -> list[str]:
    """Validate manifest records, returning a list of error messages.

    Checks:
    - All required fields present.
    - ``id`` uniqueness.
    - ``difficulty`` in range 1..5.
    - ``culture_tags`` is a list.
    - If *vocab* provided: all tags are in the vocabulary.

    Args:
        records: Loaded manifest records.
        vocab: Optional :class:`~formosa_dual.data.tag_vocab.TagVocabulary`
            for tag membership check.

    Returns:
        List of error message strings.  Empty means valid.
    """
    errors: list[str] = []
    seen_ids: set[str] = set()

    for idx, rec in enumerate(records):
        prefix = f"Record[{idx}] id={rec.get('id', '<missing>')}"

        # Required fields
        for field in _REQUIRED_FIELDS:
            if field not in rec:
                errors.append(f"{prefix}: missing required field '{field}'")

        # ID uniqueness
        rec_id = rec.get("id")
        if rec_id is not None:
            if rec_id in seen_ids:
                errors.append(f"{prefix}: duplicate id '{rec_id}'")
            seen_ids.add(rec_id)

        # Difficulty range
        difficulty = rec.get("difficulty")
        if difficulty is not None and not (1 <= difficulty <= 5):
            errors.append(f"{prefix}: difficulty={difficulty} not in 1..5")

        # culture_tags type
        tags = rec.get("culture_tags")
        if tags is not None and not isinstance(tags, list):
            errors.append(f"{prefix}: culture_tags must be a list")

        # Vocab membership
        if vocab is not None and isinstance(tags, list):
            for tag in tags:
                if tag not in vocab:
                    errors.append(f"{prefix}: tag '{tag}' not in vocabulary")

    return errors
