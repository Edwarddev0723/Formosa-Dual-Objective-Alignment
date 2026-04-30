"""Integration tests for the FormosaDataset + DualCollator pipeline (§8.3).

These tests use synthetic data and a mock processor to avoid downloading
the full Qwen2.5-VL backbone.
"""
from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from formosa_dual.data.collator import DualCollator
from formosa_dual.data.dataset import FormosaDataset
from formosa_dual.data.negative_sampler import NegativeSampler
from formosa_dual.data.tag_vocab import TagVocabulary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    """Create a tiny synthetic dataset directory."""
    from formosa_dual.utils.synthetic import make_synthetic_record

    out = tmp_path_factory.mktemp("synth_pipeline")
    img_dir = out / "images"
    img_dir.mkdir()

    tags_raw = [
        {"id": i, "tag": f"tag_{i}", "tier": 1, "freq": max(1, 10 - i), "category": "test"}
        for i in range(8)
    ]
    vocab_path = out / "vocab.json"
    vocab_path.write_text(
        json.dumps({"version": "v1", "size": len(tags_raw), "tags": tags_raw}),
        encoding="utf-8",
    )

    vocab_tags = [e["tag"] for e in tags_raw]
    records = [make_synthetic_record(i, vocab_tags, image_dir=img_dir) for i in range(4)]

    manifest_path = out / "manifest.jsonl"
    with manifest_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    return out


@pytest.fixture(scope="module")
def vocab(synth_dir):
    return TagVocabulary(synth_dir / "vocab.json")


@pytest.fixture(scope="module")
def dataset(synth_dir, vocab):
    return FormosaDataset(
        manifest_path=synth_dir / "manifest.jsonl",
        vocab=vocab,
        image_root=synth_dir,
    )


def _make_mock_processor(vocab_size: int = 32, seq_len: int = 16, batch_size: int = 2):
    """Return a minimal mock that mimics the Qwen processor API."""
    class MockProcessor:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
            return "mock_text"

        def __call__(self, text, images, padding, truncation, return_tensors):
            self.calls.append(
                {
                    "text": text,
                    "padding": padding,
                    "truncation": truncation,
                    "return_tensors": return_tensors,
                }
            )
            B = len(text)
            return {
                "input_ids": torch.ones(B, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(B, seq_len, dtype=torch.long),
                "pixel_values": torch.zeros(B, 3, 224, 224),
            }

        class tokenizer:
            @staticmethod
            def convert_tokens_to_ids(token):
                return 1

            @staticmethod
            def __call__(text, add_special_tokens=False, truncation=True, max_length=None):
                ids = list(range(len(text)))
                if truncation and max_length is not None:
                    ids = ids[:max_length]
                return {"input_ids": ids}

            @staticmethod
            def decode(token_ids, skip_special_tokens=True):
                return "x" * len(token_ids)

            pad_token_id = 0

    return MockProcessor()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dataset_yields_expected_keys(dataset):
    item = dataset[0]
    for key in ("id", "image", "caption", "pos_tag_ids", "difficulty", "source", "metadata"):
        assert key in item, f"Missing key: {key}"
    assert isinstance(item["image"], Image.Image)
    assert isinstance(item["pos_tag_ids"], list)


def test_collator_pads_correctly(dataset, vocab):
    processor = _make_mock_processor()
    sampler = NegativeSampler(vocab, strategy="uniform", num_negatives=4, seed=0)
    collator = DualCollator(processor, vocab, sampler, max_caption_tokens=16, max_pos_tags=5)

    batch = [dataset[i] for i in range(2)]
    out = collator(batch)

    assert "input_ids" in out
    assert "labels" in out
    assert out["pos_tag_ids"].shape == (2, 5)
    assert out["pos_tag_mask"].dtype == torch.bool
    assert out["pos_tag_ids"].shape[0] == 2
    assert processor.calls[-1]["truncation"] is False


def test_collator_negative_sampling_works(dataset, vocab):
    processor = _make_mock_processor()
    sampler = NegativeSampler(vocab, strategy="uniform", num_negatives=4, seed=0)
    collator = DualCollator(processor, vocab, sampler, max_caption_tokens=16, max_pos_tags=5)

    batch = [dataset[i] for i in range(2)]
    out = collator(batch)

    assert "neg_tag_ids" in out
    B, M = out["neg_tag_ids"].shape
    assert B == 2
    assert M > 0
