"""tests/unit/test_culturalness.py — required by spec §8.2.

Required tests:
    - test_f1_tag_perfect_match
    - test_f1_tag_no_overlap
    - test_s_idf_higher_for_specific_terms
    - test_composite_in_unit_range
    - test_sensitivity_analysis_returns_robustness

The NLI scorer is mocked so this file does not load the 330M Erlangshen model.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def vocab(tmp_path):
    """Tiny TagVocabulary with culturally-marked Chinese tags."""
    from formosa_dual.data.tag_vocab import TagVocabulary
    tag_strings = [
        "媽祖", "廟宇", "夜市", "山峰", "河流",
        "燈籠", "古蹟", "板橋", "清代", "原住民",
    ]
    vocab_obj = {
        "version": "v1",
        "size": len(tag_strings),
        "tags": [
            {"id": i, "tag": t, "tier": 1, "freq": 100, "category": "test"}
            for i, t in enumerate(tag_strings)
        ],
    }
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(vocab_obj, ensure_ascii=False), encoding="utf-8")
    return TagVocabulary(path)


@pytest.fixture
def idf_corpus(tmp_path):
    """Small IDF corpus where '媽祖' is rare and '圖片' is common."""
    path = tmp_path / "idf.jsonl"
    captions = (
        ["這 是 一 張 圖片"] * 9          # "圖片" appears in every doc → low IDF
        + ["媽祖 廟宇 燈籠"]              # "媽祖" / "燈籠" appear once → high IDF
    )
    with path.open("w", encoding="utf-8") as fh:
        for c in captions:
            fh.write(json.dumps({"caption": c}, ensure_ascii=False) + "\n")
    return path


@pytest.fixture
def scorer(vocab, idf_corpus):
    """CulturalnessAuto with the NLI scorer patched out."""
    with patch("formosa_dual.eval.nli_factuality.NLIFactualityScorer") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_inst.score.return_value = 0.5
        from formosa_dual.eval.culturalness import CulturalnessAuto
        instance = CulturalnessAuto(
            vocab=vocab,
            idf_corpus_path=idf_corpus,
            nli_model="mock-nli",
            weights=(0.40, 0.30, 0.30),
        )
        # Force a deterministic NLI result for all tests.
        instance._nli = mock_inst
        yield instance


def test_f1_tag_perfect_match(scorer):
    """F1 == 1.0 when generated mentions exactly the reference tags."""
    generated = "媽祖廟宇"
    reference = {
        "caption": "媽祖廟宇",
        "culture_tags": ["媽祖", "廟宇"],
        "metadata": {"article_title": "媽祖", "ocr_text": "廟宇"},
    }
    f1 = scorer._compute_f1_tag(generated, reference["culture_tags"])
    assert f1 == 1.0


def test_f1_tag_no_overlap(scorer):
    """F1 == 0.0 when predicted and reference tag sets are disjoint."""
    generated = "山峰河流"
    reference = {
        "caption": "山峰河流",
        "culture_tags": ["媽祖", "廟宇"],
        "metadata": {"article_title": "", "ocr_text": ""},
    }
    f1 = scorer._compute_f1_tag(generated, reference["culture_tags"])
    assert f1 == 0.0


def test_s_idf_higher_for_specific_terms(scorer):
    """S_IDF score is strictly higher for a culturally-specific caption."""
    common_caption = "圖片 圖片"
    specific_caption = "媽祖 燈籠"
    s_common = scorer._compute_s_idf(common_caption)
    s_specific = scorer._compute_s_idf(specific_caption)
    assert s_specific > s_common


def test_composite_in_unit_range(scorer):
    """The composite score is bounded in [0, 1] for a typical reference."""
    generated = "媽祖廟"
    reference = {
        "caption": "媽祖廟燈籠",
        "culture_tags": ["媽祖", "廟宇"],
        "metadata": {"article_title": "媽祖", "ocr_text": "廟宇"},
    }
    out = scorer.score(generated, reference)
    for k in ("F1_tag", "S_IDF", "E_NLI", "composite"):
        assert k in out
    # Each individual sub-score must lie in [0, 1] under the mocked NLI.
    assert 0.0 <= out["F1_tag"] <= 1.0
    assert 0.0 <= out["E_NLI"] <= 1.0
    assert 0.0 <= out["composite"] <= 1.0 + 1e-6


def test_sensitivity_analysis_returns_robustness(scorer):
    """Sensitivity analysis returns weight samples and per-weight composites."""
    generated = ["媽祖廟", "山峰河流"]
    references = [
        {
            "caption": "媽祖廟",
            "culture_tags": ["媽祖", "廟宇"],
            "metadata": {"article_title": "媽祖", "ocr_text": "廟宇"},
        },
        {
            "caption": "山峰河流",
            "culture_tags": ["山峰", "河流"],
            "metadata": {"article_title": "山峰", "ocr_text": ""},
        },
    ]
    result = scorer.sensitivity_analysis(generated, references, n_samples=5)
    assert "weight_samples" in result
    assert "composite_scores_per_weight" in result
    assert len(result["weight_samples"]) == 5
    assert len(result["composite_scores_per_weight"]) == 5
    for entry in result["composite_scores_per_weight"]:
        assert "weights" in entry
        assert "composites" in entry
        assert len(entry["composites"]) == len(generated)
