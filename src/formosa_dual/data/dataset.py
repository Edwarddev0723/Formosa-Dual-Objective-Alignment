"""formosa_dual.data.dataset — PyTorch Dataset for Formosa manifests.

Returns per-item dicts that :class:`~formosa_dual.data.collator.DualCollator`
processes into model-ready batches.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from formosa_dual.data.manifest import load_manifest
from formosa_dual.data.tag_vocab import TagVocabulary
from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


class FormosaDataset(torch.utils.data.Dataset):
    """PyTorch Dataset over a Formosa JSONL manifest.

    Each item returns a dict with keys:
        ``id``, ``image``, ``caption``, ``pos_tag_ids``, ``difficulty``,
        ``source``, ``metadata``.

    Args:
        manifest_path: Path to the ``.jsonl`` manifest file.
        vocab: Vocabulary for encoding tag strings to integer ids.
        image_root: Root directory prepended to relative ``image_path`` values.
        smoke_max_samples: If set, truncate dataset to this many samples.
        difficulty_filter: ``(min, max)`` inclusive difficulty range filter.
    """

    def __init__(
        self,
        manifest_path: Path,
        vocab: TagVocabulary,
        image_root: Path,
        smoke_max_samples: int | None = None,
        difficulty_filter: tuple[int, int] | None = None,
    ) -> None:
        self._vocab = vocab
        self._image_root = Path(image_root)

        records = load_manifest(Path(manifest_path))

        # Apply difficulty filter
        if difficulty_filter is not None:
            lo, hi = difficulty_filter
            records = [r for r in records if lo <= r.get("difficulty", 1) <= hi]

        # Smoke truncation
        if smoke_max_samples is not None:
            records = records[:smoke_max_samples]

        self._records = records
        logger.info(
            "FormosaDataset: %d samples (manifest=%s)", len(records), manifest_path
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        rec = self._records[idx]

        # Resolve image path
        img_path_str = rec.get("image_path", "")
        img_path = Path(img_path_str)
        if not img_path.is_absolute() and not img_path.exists():
            # Treat manifest path as relative to image_root only when it doesn't
            # already resolve from CWD (some manifests store repo-relative paths).
            img_path = self._image_root / img_path

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError) as exc:
            raise FileNotFoundError(
                f"Image not found for record '{rec.get('id')}': {img_path.resolve()}"
            ) from exc

        # Encode positive tags (skip unknown tags gracefully)
        pos_tag_ids: list[int] = []
        for tag_str in rec.get("culture_tags") or []:
            tid = self._vocab.encode(tag_str)
            if tid is not None:
                pos_tag_ids.append(tid)

        return {
            "id": rec.get("id", ""),
            "image": image,
            "caption": rec.get("caption", ""),
            "pos_tag_ids": pos_tag_ids,
            "difficulty": rec.get("difficulty", 1),
            "source": rec.get("source", ""),
            "metadata": rec.get("metadata") or {},
        }
