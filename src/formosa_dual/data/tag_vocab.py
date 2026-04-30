"""formosa_dual.data.tag_vocab — tag vocabulary management.

The vocabulary JSON schema (§5.5):
    {
        "version": "v1",
        "size": 800,
        "tags": [
            {"id": 0, "tag": "媽祖", "tier": 1, "freq": 230, "category": "宗教"},
            ...
        ]
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


class TagVocabulary:
    """Tag vocabulary loaded from a JSON file conforming to the spec schema."""

    def __init__(self, vocab_path: Path) -> None:
        vocab_path = Path(vocab_path)
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_path.resolve()}")

        with vocab_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        self._version: str = data.get("version", "v1")
        self._tag_entries: list[dict] = data.get("tags", [])

        # Build lookup maps
        self._tag_to_id: dict[str, int] = {}
        self._id_to_tag: dict[int, str] = {}
        self._id_to_category: dict[int, str] = {}
        self._id_to_freq: dict[int, int] = {}

        for entry in self._tag_entries:
            tid = int(entry["id"])
            tag = entry["tag"]
            self._tag_to_id[tag] = tid
            self._id_to_tag[tid] = tag
            self._id_to_category[tid] = entry.get("category", "")
            self._id_to_freq[tid] = int(entry.get("freq", 0))

        logger.info("Loaded TagVocabulary: %d tags (version=%s)", len(self._tag_entries), self._version)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode(self, tag_str: str) -> int | None:
        """Return the integer id for *tag_str*, or ``None`` if not in vocab."""
        return self._tag_to_id.get(tag_str)

    def decode(self, tag_id: int) -> str:
        """Return the tag string for *tag_id*.

        Raises:
            KeyError: If *tag_id* is not in the vocabulary.
        """
        if tag_id not in self._id_to_tag:
            raise KeyError(f"Tag id {tag_id} not in vocabulary")
        return self._id_to_tag[tag_id]

    def category_of(self, tag_id: int) -> str:
        """Return the category string for *tag_id*."""
        if tag_id not in self._id_to_category:
            raise KeyError(f"Tag id {tag_id} not in vocabulary")
        return self._id_to_category[tag_id]

    def freq_of(self, tag_id: int) -> int:
        """Return the corpus frequency for *tag_id*."""
        if tag_id not in self._id_to_freq:
            raise KeyError(f"Tag id {tag_id} not in vocabulary")
        return self._id_to_freq[tag_id]

    def __len__(self) -> int:
        return len(self._tag_entries)

    def __contains__(self, tag_str: object) -> bool:
        return tag_str in self._tag_to_id

    def __iter__(self) -> Iterator[str]:
        return iter(self._tag_to_id)

    @property
    def tags(self) -> list[str]:
        """Ordered list of all tag strings (ordered by id)."""
        return [e["tag"] for e in sorted(self._tag_entries, key=lambda e: e["id"])]

    # ------------------------------------------------------------------
    # Build + save
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        tier1: list[str],
        tier2: list[str],
        tier3: list[str],
        freqs: dict[str, int],
        categories: dict[str, str],
        target_size: int = 800,
        min_freq: int = 5,
    ) -> "TagVocabulary":
        """Build a new TagVocabulary from raw tag lists.

        Priority: tier1 (all included) > tier2 > tier3 (filtered by min_freq).
        Truncate combined list to *target_size*.

        Args:
            tier1: High-priority tags (always included).
            tier2: Medium-priority tags.
            tier3: Low-priority tags (filtered by *min_freq*).
            freqs: Mapping of tag → corpus frequency.
            categories: Mapping of tag → category string.
            target_size: Maximum vocabulary size.
            min_freq: Minimum frequency for tier3 tags.

        Returns:
            New :class:`TagVocabulary` instance (not yet saved to disk).
        """
        import tempfile

        seen: set[str] = set()
        entries: list[dict] = []

        def _add(tag: str, tier: int) -> None:
            if tag in seen:
                return
            seen.add(tag)
            entries.append({
                "id": len(entries),
                "tag": tag,
                "tier": tier,
                "freq": freqs.get(tag, 0),
                "category": categories.get(tag, ""),
            })

        for t in tier1:
            _add(t, 1)
        for t in tier2:
            _add(t, 2)
        for t in tier3:
            if freqs.get(t, 0) >= min_freq:
                _add(t, 3)

        entries = entries[:target_size]

        vocab_data = {"version": "v1", "size": len(entries), "tags": entries}

        # Serialize to a temp file and load via normal path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                         encoding="utf-8") as tmp:
            json.dump(vocab_data, tmp, ensure_ascii=False)
            tmp_path = Path(tmp.name)

        instance = cls(tmp_path)
        tmp_path.unlink(missing_ok=True)
        return instance

    def save(self, path: Path) -> None:
        """Serialize this vocabulary to *path* in the spec JSON format.

        Args:
            path: Destination ``.json`` file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        vocab_data = {
            "version": self._version,
            "size": len(self._tag_entries),
            "tags": self._tag_entries,
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(vocab_data, fh, ensure_ascii=False, indent=2)
        logger.info("Saved TagVocabulary (%d tags) to %s", len(self._tag_entries), path)
