"""tests/unit/test_dataset_image_resolution.py

Regression tests for `FormosaDataset` image-path resolution.

Catches the bug seen during DoD §0.6 step-5: when a manifest stores
repo-relative paths (e.g., ``data/synthetic/images/foo.png``) AND
``image_root`` is ``data/synthetic/images``, the old code unconditionally
joined the two and produced ``data/synthetic/images/data/synthetic/images/foo.png``.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from formosa_dual.data.dataset import FormosaDataset
from formosa_dual.data.tag_vocab import TagVocabulary


def _write_vocab(path: Path) -> TagVocabulary:
    payload = {
        "version": "v1",
        "tags": [
            {"id": 0, "tag": "媽祖", "category": "religion", "frequency": 1},
            {"id": 1, "tag": "夜市", "category": "lifestyle", "frequency": 1},
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return TagVocabulary(path)


def _write_manifest(records, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(path)


def test_manifest_path_already_resolves_from_cwd(tmp_path, monkeypatch):
    """When manifest stores a path resolvable from CWD, do not double-prefix."""
    monkeypatch.chdir(tmp_path)
    images = tmp_path / "data/synthetic/images"
    img_path = images / "synth_0.png"
    _make_image(img_path)

    vocab = _write_vocab(tmp_path / "vocab.json")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        [{
            "id": "x0",
            "image_path": "data/synthetic/images/synth_0.png",  # resolves from CWD
            "caption": "test",
            "culture_tags": ["媽祖"],
            "difficulty": 1,
        }],
        manifest,
    )

    ds = FormosaDataset(manifest, vocab, image_root=images)
    item = ds[0]
    assert item["image"].size == (32, 32)
    assert item["pos_tag_ids"] == [0]


def test_manifest_path_relative_to_image_root(tmp_path, monkeypatch):
    """When manifest stores image_root-relative path, prepend image_root."""
    monkeypatch.chdir(tmp_path)
    images = tmp_path / "imgs"
    img_path = images / "a.png"
    _make_image(img_path)

    vocab = _write_vocab(tmp_path / "vocab.json")
    manifest = tmp_path / "m.jsonl"
    _write_manifest(
        [{
            "id": "x1",
            "image_path": "a.png",  # relative to image_root
            "caption": "test",
            "culture_tags": [],
            "difficulty": 1,
        }],
        manifest,
    )

    ds = FormosaDataset(manifest, vocab, image_root=images)
    item = ds[0]
    assert item["image"].size == (32, 32)
