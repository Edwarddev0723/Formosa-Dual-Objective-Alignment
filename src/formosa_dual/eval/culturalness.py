"""formosa_dual.eval.culturalness — composite culturalness metric (§5.21).

    Culturalness_auto = 0.40 * F1_tag + 0.30 * S_IDF + 0.30 * E_NLI
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from formosa_dual.data.tag_annotator import TagAnnotator
from formosa_dual.data.tag_vocab import TagVocabulary
from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


class CulturalnessAuto:
    """Composite cultural faithfulness metric.

    Args:
        vocab: :class:`~formosa_dual.data.tag_vocab.TagVocabulary`.
        idf_corpus_path: Path to a JSONL file of captions for IDF computation
            (each line: ``{"caption": "..."}``).
        nli_model: HuggingFace NLI model ID.
        weights: Tuple ``(w_f1, w_idf, w_nli)`` — must sum to 1.
    """

    def __init__(
        self,
        vocab: TagVocabulary,
        idf_corpus_path: Path,
        nli_model: str = "IDEA-CCNL/Erlangshen-Roberta-330M-NLI",
        weights: tuple[float, float, float] = (0.40, 0.30, 0.30),
    ) -> None:
        self.vocab = vocab
        self.weights = weights
        self._annotator = TagAnnotator(vocab=vocab)
        self._idf = self._build_idf(vocab, Path(idf_corpus_path))

        from formosa_dual.eval.nli_factuality import NLIFactualityScorer
        self._nli = NLIFactualityScorer(nli_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, generated: str, reference: dict) -> dict:
        """Score a single generated caption.

        Args:
            generated: Generated caption string.
            reference: Dict with keys ``caption``, ``culture_tags``, ``metadata``
                (must have ``article_title`` and optionally ``ocr_text``).

        Returns:
            Dict: ``{F1_tag, S_IDF, E_NLI, composite}``.
        """
        f1 = self._compute_f1_tag(generated, reference.get("culture_tags", []))
        s_idf = self._compute_s_idf(generated)
        premise = self._build_premise(reference)
        e_nli = self._nli.score(generated, premise)

        w_f1, w_idf, w_nli = self.weights
        composite = w_f1 * f1 + w_idf * s_idf + w_nli * e_nli

        return {"F1_tag": f1, "S_IDF": s_idf, "E_NLI": e_nli, "composite": composite}

    def score_batch(self, generated_list: list[str], reference_list: list[dict]) -> dict:
        """Score a batch of generated captions.

        Args:
            generated_list: List of generated captions.
            reference_list: Corresponding list of reference dicts.

        Returns:
            Dict with mean scalar metrics and per-sample arrays.
        """
        results = [self.score(g, r) for g, r in zip(generated_list, reference_list)]
        keys = ["F1_tag", "S_IDF", "E_NLI", "composite"]
        agg: dict = {}
        for k in keys:
            vals = [r[k] for r in results]
            agg[k] = vals
            agg[f"mean_{k}"] = sum(vals) / max(len(vals), 1)
        return agg

    def sensitivity_analysis(
        self,
        generated_list: list[str],
        reference_list: list[dict],
        n_samples: int = 20,
    ) -> dict:
        """Sample weight triplets uniformly on the simplex and measure ranking robustness.

        Args:
            generated_list: List of generated captions.
            reference_list: Corresponding list of reference dicts.
            n_samples: Number of weight triplets to sample.

        Returns:
            Dict with ``weight_samples`` and ``composite_scores_per_weight``.
        """
        per_sample_base = [self.score(g, r) for g, r in zip(generated_list, reference_list)]
        weight_samples = _sample_simplex(n_samples)
        results = []
        for w_f1, w_idf, w_nli in weight_samples:
            composites = [
                w_f1 * s["F1_tag"] + w_idf * s["S_IDF"] + w_nli * s["E_NLI"]
                for s in per_sample_base
            ]
            results.append({"weights": (w_f1, w_idf, w_nli), "composites": composites})
        return {"weight_samples": weight_samples, "composite_scores_per_weight": results}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_f1_tag(self, generated: str, reference_tags: list[str]) -> float:
        """Compute tag-level F1."""
        record = {"caption": generated, "id": "__eval__", "metadata": {}}
        # TagAnnotator.annotate returns vocab tag strings (spec §5.6).
        predicted_tags = set(self._annotator.annotate(record))
        ref_set = set(reference_tags)
        if not ref_set and not predicted_tags:
            return 1.0
        if not ref_set or not predicted_tags:
            return 0.0
        tp = len(predicted_tags & ref_set)
        precision = tp / len(predicted_tags)
        recall = tp / len(ref_set)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _compute_s_idf(self, generated: str) -> float:
        """Compute IDF-weighted cultural score."""
        tokens = _simple_tokenize(generated)
        if not tokens:
            return 0.0
        total = 0.0
        for tok in tokens:
            total += self._idf.get(tok, 0.0)
        return total / len(tokens)

    @staticmethod
    def _build_premise(reference: dict) -> str:
        meta = reference.get("metadata", {})
        title = meta.get("article_title", "")
        ocr = meta.get("ocr_text", "")
        return (title + " " + ocr).strip()

    @staticmethod
    def _build_idf(vocab: TagVocabulary, corpus_path: Path) -> dict[str, float]:
        """Compute IDF for vocab tokens over a JSONL corpus."""
        if not corpus_path.exists():
            logger.warning("IDF corpus not found at %s; S_IDF will be 0.", corpus_path)
            return {}
        doc_count: dict[str, int] = {}
        n_docs = 0
        with corpus_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                caption = rec.get("caption", "")
                tokens = set(_simple_tokenize(caption))
                for tok in tokens:
                    doc_count[tok] = doc_count.get(tok, 0) + 1
                n_docs += 1
        idf: dict[str, float] = {}
        for tok, df in doc_count.items():
            idf[tok] = math.log((n_docs + 1) / (df + 1))
        return idf


def _simple_tokenize(text: str) -> list[str]:
    """Very simple tokenizer: split on whitespace and punctuation."""
    import re
    return re.findall(r"[\w\u4e00-\u9fff]+", text)


def _sample_simplex(n: int) -> list[tuple[float, float, float]]:
    """Sample *n* points uniformly on the 2-simplex."""
    points = []
    for _ in range(n):
        u = random.random()
        v = random.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        w = 1 - u - v
        points.append((u, v, w))
    return points
