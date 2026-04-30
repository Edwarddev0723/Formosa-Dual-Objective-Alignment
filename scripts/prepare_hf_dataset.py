#!/usr/bin/env python
"""Convert a Hugging Face image-caption dataset into Formosa-Dual manifests.

Example:
    python scripts/prepare_hf_dataset.py \
        --dataset renhehuang/formosa-vlm-caption-v1 \
        --split train \
        --output-dir data/raw
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile

from formosa_dual.utils.logging import get_logger

logger = get_logger("prepare_hf_dataset")


_IMAGE_CANDIDATES = ("images", "image", "img", "picture", "pil_image")
_CAPTION_CANDIDATES = (
    "caption",
    "text",
    "Finegrained_text",
    "Fine-grained text",
    "finegrained_text",
    "fine_grained_text",
    "description",
    "answer",
    "response",
)
_ID_CANDIDATES = ("id", "uuid", "image_id", "sample_id")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a Hugging Face image-caption dataset for Formosa-Dual."
    )
    parser.add_argument("--dataset", required=True, help="HF dataset id.")
    parser.add_argument("--name", default=None, help="Optional HF dataset config name.")
    parser.add_argument("--split", default="train", help="HF split to load.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--caption-column", default="auto")
    parser.add_argument("--id-column", default="auto")
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        from datasets import Image as HfImage
        from datasets import load_dataset
    except ImportError:
        logger.error("Missing dependency: datasets. Install requirements/base.txt first.")
        sys.exit(2)

    output_dir = args.output_dir
    image_dir = args.image_dir or output_dir / "images"
    manifest_path = args.manifest or output_dir / "manifest.jsonl"
    captions_path = output_dir / "captions.txt"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading HF dataset %s split=%s", args.dataset, args.split)
    ds_kwargs: dict[str, Any] = {"split": args.split}
    if args.name is not None:
        ds_kwargs["name"] = args.name
    dataset = load_dataset(args.dataset, **ds_kwargs)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    columns = list(dataset.column_names)
    image_column = _resolve_column(args.image_column, columns, _IMAGE_CANDIDATES, "image")
    caption_column = _resolve_column(
        args.caption_column, columns, _CAPTION_CANDIDATES, "caption", required=False
    )
    id_column = _resolve_column(args.id_column, columns, _ID_CANDIDATES, "id", required=False)

    logger.info(
        "Resolved columns: image=%s caption=%s id=%s all=%s",
        image_column,
        caption_column,
        id_column,
        columns,
    )

    try:
        dataset = dataset.cast_column(image_column, HfImage(decode=False))
        logger.info("Using decode=False for image column; images will be validated by this script.")
    except Exception as exc:
        logger.warning("Could not disable automatic image decoding for %s: %s", image_column, exc)

    records: list[dict[str, Any]] = []
    skipped = 0
    id_counts: dict[str, int] = {}
    for idx in range(len(dataset)):
        try:
            row = dataset[idx]
            raw_id = str(row.get(id_column) if id_column else f"sample_{idx:06d}")
            sample_id = _dedupe_id(_clean_id(raw_id), id_counts)
            caption = _extract_caption(row, caption_column)
            image = _extract_image(row.get(image_column))

            img_path, sha256, phash, width, height = _save_image(image, image_dir, sample_id)
            source_value = _source_value(row, args.source_name or args.dataset)

            records.append(
                {
                    "id": sample_id,
                    "image_path": str(img_path),
                    "caption": caption,
                    "source": source_value,
                    "article_url": _article_url(row),
                    "image_hash": f"sha256:{sha256}",
                    "phash": phash,
                    "width": width,
                    "height": height,
                    "difficulty": _difficulty(row),
                    "culture_tags": _culture_tags(row),
                    "metadata": _metadata(row, skip={image_column}),
                }
            )
        except Exception as exc:
            skipped += 1
            logger.warning("Skipping sample index=%d: %s", idx, exc)
            continue
        if (idx + 1) % 100 == 0:
            logger.info("Scanned %d samples; prepared=%d skipped=%d", idx + 1, len(records), skipped)

    with manifest_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    captions_path.write_text(
        "\n".join(record["caption"] for record in records if record["caption"]) + "\n",
        encoding="utf-8",
    )

    logger.info("Wrote %d records to %s", len(records), manifest_path)
    if skipped:
        logger.warning("Skipped %d samples with unreadable images or malformed records.", skipped)
    logger.info("Wrote captions to %s", captions_path)
    logger.info("Images written to %s", image_dir)


def _resolve_column(
    requested: str,
    columns: list[str],
    candidates: tuple[str, ...],
    label: str,
    required: bool = True,
) -> str | None:
    if requested != "auto":
        if requested not in columns:
            raise SystemExit(f"{label} column not found: {requested}. Available: {columns}")
        return requested
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if required:
        raise SystemExit(f"Could not auto-detect {label} column. Available: {columns}")
    return None


def _clean_id(value: str) -> str:
    value = value.strip() or "sample"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:160]


def _dedupe_id(value: str, counts: dict[str, int]) -> str:
    count = counts.get(value, 0)
    counts[value] = count + 1
    if count == 0:
        return value
    return f"{value}_{count}"


def _extract_caption(row: dict[str, Any], caption_column: str | None) -> str:
    if caption_column and row.get(caption_column) is not None:
        value = row[caption_column]
        if isinstance(value, str):
            return value.strip()
    for field in ("messages", "conversations"):
        value = row.get(field)
        if isinstance(value, list):
            caption = _caption_from_messages(value)
            if caption:
                return caption
    return ""


def _caption_from_messages(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or msg.get("from") or "").lower()
        text = msg.get("content", msg.get("value", ""))
        if isinstance(text, list):
            text = " ".join(str(part.get("text", "")) for part in text if isinstance(part, dict))
        if role in {"assistant", "gpt", "model"} and isinstance(text, str) and text.strip():
            return text.replace("<image>", "").strip()
    return ""


def _extract_image(value: Any) -> Image.Image:
    if isinstance(value, list):
        if not value:
            raise ValueError("Image list is empty")
        return _extract_image(value[0])
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") is not None:
            return _extract_image(value["path"])
    if isinstance(value, (str, Path)):
        value_str = str(value)
        if value_str.startswith(("http://", "https://")):
            with urllib.request.urlopen(value_str, timeout=30) as resp:  # noqa: S310
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        return Image.open(value_str).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def _save_image(image: Image.Image, image_dir: Path, sample_id: str) -> tuple[Path, str, str, int, int]:
    image = image.convert("RGB")
    output_path = image_dir / f"{sample_id}.jpg"

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    data = buf.getvalue()
    output_path.write_bytes(data)

    sha256 = hashlib.sha256(data).hexdigest()
    phash = ""
    try:
        import imagehash

        phash = str(imagehash.phash(image))
    except Exception:
        pass
    width, height = image.size
    return output_path, sha256, phash, width, height


def _source_value(row: dict[str, Any], fallback: str) -> str:
    value = row.get("source")
    return value if isinstance(value, str) and value else fallback


def _article_url(row: dict[str, Any]) -> str:
    for key in ("article_url", "url", "source_url"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    source = row.get("source")
    return source if isinstance(source, str) and source.startswith(("http://", "https://")) else ""


def _difficulty(row: dict[str, Any]) -> int:
    value = row.get("difficulty", 3)
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 3
    return min(max(value, 1), 5)


def _culture_tags(row: dict[str, Any]) -> list[str]:
    for key in ("culture_tags", "tags", "entities", "keywords"):
        value = row.get(key)
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [part.strip() for part in re.split(r"[,，;；|]", value) if part.strip()]
    return []


def _metadata(row: dict[str, Any], skip: set[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in row.items():
        if key in skip:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
    return metadata


if __name__ == "__main__":
    main()
