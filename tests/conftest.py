"""tests/conftest.py — shared fixtures (spec §8.1).

Required fixtures:
    - synthetic_data_dir : session-scoped synthetic data directory
    - device             : auto-detected torch device for tests
    - tiny_vocab         : 16-tag synthetic TagVocabulary
    - tiny_dataset       : 8-sample synthetic FormosaDataset
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory) -> Path:
    """Build synthetic data once per test session.

    Creates ``train_synth.jsonl``, ``val_synth.jsonl``, ``vocab_synth.json``
    and an ``images/`` directory under a temporary path.
    """
    from formosa_dual.utils.synthetic import make_synthetic_record

    data_dir = tmp_path_factory.mktemp("synthetic")
    image_dir = data_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # 16-tag vocab in TagVocabulary JSON schema
    tag_strings = [
        "媽祖", "廟宇", "夜市", "山峰", "河流",
        "市集", "燈籠", "橋梁", "古蹟", "板橋",
        "清代", "原住民", "茶園", "稻田", "海岸",
        "捷運",
    ]
    vocab_obj = {
        "version": "v1",
        "size": len(tag_strings),
        "tags": [
            {"id": i, "tag": t, "tier": 1, "freq": 100, "category": "test"}
            for i, t in enumerate(tag_strings)
        ],
    }
    vocab_path = data_dir / "vocab_synth.json"
    vocab_path.write_text(json.dumps(vocab_obj, ensure_ascii=False), encoding="utf-8")

    # Train + val manifests
    def _write_split(name: str, n: int, start: int) -> Path:
        path = data_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for i in range(start, start + n):
                rec = make_synthetic_record(i, tag_strings, image_dir=image_dir)
                # rewrite image_path to be relative to data_dir for portability
                rec["image_path"] = str(Path(rec["image_path"]).resolve())
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path

    _write_split("train_synth", n=8, start=0)
    _write_split("val_synth", n=4, start=100)

    return data_dir


@pytest.fixture
def device() -> torch.device:
    """Auto-detect a torch device for tests (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def tiny_vocab(synthetic_data_dir):
    """Load the 16-tag synthetic TagVocabulary."""
    from formosa_dual.data.tag_vocab import TagVocabulary
    return TagVocabulary(synthetic_data_dir / "vocab_synth.json")


@pytest.fixture
def tiny_dataset(synthetic_data_dir, tiny_vocab):
    """Build the 8-sample synthetic FormosaDataset (train split)."""
    from formosa_dual.data.dataset import FormosaDataset
    return FormosaDataset(
        manifest_path=synthetic_data_dir / "train_synth.jsonl",
        vocab=tiny_vocab,
        image_root=synthetic_data_dir / "images",
    )
