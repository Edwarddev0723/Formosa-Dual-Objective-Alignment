"""formosa_dual.utils.synthetic — lightweight synthetic data primitives.

Used by ``scripts/make_synthetic_data.py`` and test fixtures.
Must run in <10 seconds on Mac.
"""
from __future__ import annotations

import hashlib
import random

from PIL import Image

# Colour palette for deterministic image generation
_COLORS = [
    "紅色", "藍色", "綠色", "黃色", "白色",
    "黑色", "橙色", "紫色", "粉紅色", "青色",
]
_OBJECTS = [
    "廟宇", "山峰", "河流", "市集", "建築物",
    "花朵", "動物", "人物", "燈籠", "橋梁",
]


def make_synthetic_image(rgb_seed: int, size: tuple[int, int] = (224, 224)) -> Image.Image:
    """Create a deterministic solid-colour PNG image from *rgb_seed*.

    Args:
        rgb_seed: Integer seed; determines the pixel colour.
        size: ``(width, height)`` of the image.

    Returns:
        RGB :class:`PIL.Image.Image`.
    """
    rng = random.Random(rgb_seed)
    r = rng.randint(0, 255)
    g = rng.randint(0, 255)
    b = rng.randint(0, 255)
    return Image.new("RGB", size, color=(r, g, b))


def make_synthetic_caption(template: str = "這是一張{color}的{object}圖片", fill: dict | None = None) -> str:
    """Fill *template* with provided or default values.

    Args:
        template: Python format-string template with ``{color}`` and ``{object}``
            placeholders.
        fill: Optional dict of placeholder overrides.

    Returns:
        Filled caption string.
    """
    if fill is None:
        fill = {}
    return template.format(
        color=fill.get("color", "藍色"),
        object=fill.get("object", "廟宇"),
    )


def make_synthetic_record(
    idx: int,
    vocab_tags: list[str],
    image_dir: "Path | None" = None,
    tags_per_image: int = 3,
) -> dict:
    """Build a single synthetic manifest record.

    Args:
        idx: Record index (determines seeds).
        vocab_tags: Full tag list to sample positives from.
        image_dir: If provided, save a PNG and include a relative ``image_path``.
        tags_per_image: Number of positive tags per record.

    Returns:
        Manifest record dict.
    """
    from pathlib import Path

    rng = random.Random(idx)
    color = rng.choice(_COLORS)
    obj = rng.choice(_OBJECTS)
    caption = make_synthetic_caption(fill={"color": color, "object": obj})
    tags = rng.sample(vocab_tags, min(tags_per_image, len(vocab_tags)))

    img_filename = f"synth_{idx:04d}.png"
    if image_dir is not None:
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
        img = make_synthetic_image(idx)
        img_path = image_dir / img_filename
        img.save(img_path)
        image_path_str = str(img_path)
        # Compute sha256 hash
        sha256 = hashlib.sha256(img_path.read_bytes()).hexdigest()
        phash_str = format(rng.getrandbits(64), "016x")
    else:
        image_path_str = f"images/{img_filename}"
        sha256 = "sha256:" + "0" * 64
        phash_str = format(rng.getrandbits(64), "016x")

    return {
        "id": f"synth_{idx:06d}",
        "image_path": image_path_str,
        "caption": caption,
        "source": "synthetic",
        "article_url": f"https://synthetic.example/item/{idx}",
        "image_hash": f"sha256:{sha256}",
        "phash": phash_str,
        "width": 224,
        "height": 224,
        "difficulty": rng.randint(1, 5),
        "culture_tags": tags,
        "metadata": {
            "article_title": f"合成文章{idx}",
            "ocr_text": None,
            "geo_tags": [],
            "era_tags": [],
        },
    }
