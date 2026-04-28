"""Unit tests for formosa_dual.data.tag_annotator (§8.2)."""
import pytest

from formosa_dual.data.tag_annotator import LLMClient, TagAnnotator
from formosa_dual.data.tag_vocab import TagVocabulary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_vocab(tmp_path):
    import json

    tags = [
        {"id": 0, "tag": "媽祖", "tier": 1, "freq": 50, "category": "宗教"},
        {"id": 1, "tag": "廟宇", "tier": 1, "freq": 60, "category": "建築"},
        {"id": 2, "tag": "台北", "tier": 1, "freq": 80, "category": "地理"},
        {"id": 3, "tag": "清代", "tier": 2, "freq": 25, "category": "歷史"},
        {"id": 4, "tag": "九份", "tier": 3, "freq": 10, "category": "景點"},
    ]
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"version": "v1", "size": len(tags), "tags": tags}), encoding="utf-8")
    return TagVocabulary(path)


@pytest.fixture
def annotator(tiny_vocab):
    return TagAnnotator(tiny_vocab, use_aho_corasick=True, use_metadata=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_aho_corasick_extracts_known_tags(annotator):
    record = {"caption": "台北的廟宇供奉媽祖", "metadata": {}}
    tags = annotator.annotate(record)
    assert "台北" in tags
    assert "廟宇" in tags
    assert "媽祖" in tags


def test_metadata_mapper_extracts_from_category(annotator):
    record = {
        "caption": "這是一張圖片",
        "metadata": {"article_title": "九份老街", "geo_tags": ["台北"], "era_tags": ["清代"]},
    }
    tags = annotator.annotate(record)
    assert "台北" in tags
    assert "清代" in tags


def test_max_tags_cap_respected(tiny_vocab):
    ann = TagAnnotator(tiny_vocab, use_aho_corasick=True, use_metadata=True, max_tags_per_image=2)
    record = {
        "caption": "台北的廟宇供奉媽祖，九份古蹟見清代風情",
        "metadata": {"geo_tags": ["台北"], "era_tags": ["清代"]},
    }
    tags = ann.annotate(record)
    assert len(tags) <= 2


def test_dedup_across_sources(annotator):
    # Same tag appears in both caption and metadata
    record = {
        "caption": "台北的廟宇",
        "metadata": {"geo_tags": ["台北"], "era_tags": []},
    }
    tags = annotator.annotate(record)
    assert tags.count("台北") == 1


def test_returns_only_vocab_tags(tiny_vocab):
    ann = TagAnnotator(tiny_vocab, use_aho_corasick=True, use_metadata=True)
    record = {
        "caption": "屏東縣的原住民文化",
        "metadata": {},
    }
    tags = ann.annotate(record)
    for tag in tags:
        assert tag in tiny_vocab
