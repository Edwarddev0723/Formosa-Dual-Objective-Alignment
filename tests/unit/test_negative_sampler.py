"""Unit tests for formosa_dual.data.negative_sampler (§8.2)."""
import pytest
import torch

from formosa_dual.data.negative_sampler import NegativeSampler
from formosa_dual.data.tag_vocab import TagVocabulary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_vocab(tmp_path):
    import json

    tags = [
        {"id": i, "tag": f"tag_{i}", "tier": 1, "freq": max(1, 10 - i), "category": "test"}
        for i in range(20)
    ]
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"version": "v1", "size": len(tags), "tags": tags}), encoding="utf-8")
    return TagVocabulary(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_uniform_returns_correct_count(small_vocab):
    sampler = NegativeSampler(small_vocab, strategy="uniform", num_negatives=5, seed=0)
    negs = sampler.sample(positive_ids=[0, 1])
    assert len(negs) == 5


def test_uniform_excludes_positives(small_vocab):
    sampler = NegativeSampler(small_vocab, strategy="uniform", num_negatives=10, seed=0)
    pos = [0, 1, 2]
    negs = sampler.sample(positive_ids=pos)
    for neg in negs:
        assert neg not in pos


def test_inverse_freq_prefers_rare(small_vocab):
    # Run many samples and check that rare tags (high id) appear more often
    sampler = NegativeSampler(small_vocab, strategy="inverse_freq", num_negatives=100, seed=0)
    counts: dict[int, int] = {}
    for _ in range(50):
        for tid in sampler.sample(positive_ids=[]):
            counts[tid] = counts.get(tid, 0) + 1
    # tag_19 has freq=1 (rarest after clip), tag_0 has freq=10
    rare_count = counts.get(19, 0)
    common_count = counts.get(0, 0)
    assert rare_count >= common_count  # rare should appear at least as often


def test_hard_strategy_requires_embeddings(small_vocab):
    sampler = NegativeSampler(small_vocab, strategy="hard", num_negatives=5, seed=0)
    with pytest.raises(ValueError, match="visual_emb"):
        sampler.sample(positive_ids=[0], visual_emb=None, tag_embs=None)


def test_seed_reproducibility(small_vocab):
    s1 = NegativeSampler(small_vocab, strategy="uniform", num_negatives=5, seed=42)
    s2 = NegativeSampler(small_vocab, strategy="uniform", num_negatives=5, seed=42)
    r1 = s1.sample(positive_ids=[0])
    r2 = s2.sample(positive_ids=[0])
    assert r1 == r2
