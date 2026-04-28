"""formosa_dual.eval.nli_factuality — claim splitter and NLI scorer.

Used by :class:`~formosa_dual.eval.culturalness.CulturalnessAuto` to score
factual faithfulness via natural language inference (§5.21).
"""
from __future__ import annotations

import re
from pathlib import Path

from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


class NLIFactualityScorer:
    """Score factual faithfulness of generated text using an NLI model.

    Args:
        nli_model: HuggingFace model ID for the NLI model.
            Default: ``IDEA-CCNL/Erlangshen-Roberta-330M-NLI``.

    Methods:
        score(generated, premise) -> float
            Return E_NLI: fraction of atomic claims in *generated* whose
            entailment score > contradiction score against *premise*.
    """

    def __init__(self, nli_model: str = "IDEA-CCNL/Erlangshen-Roberta-330M-NLI") -> None:
        self.nli_model = nli_model
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline
            logger.info("Loading NLI pipeline: %s", self.nli_model)
            self._pipeline = pipeline("text-classification", model=self.nli_model, top_k=None)
        return self._pipeline

    @staticmethod
    def split_claims(text: str) -> list[str]:
        """Split *text* into atomic sentence-level claims.

        Simple sentence splitter on Chinese/English punctuation.

        Args:
            text: Input text.

        Returns:
            Non-empty stripped sentences.
        """
        parts = re.split(r"[。！？.!?]", text)
        return [p.strip() for p in parts if p.strip()]

    def score(self, generated: str, premise: str) -> float:
        """Compute E_NLI: fraction of claims entailed by *premise*.

        Args:
            generated: Generated caption text.
            premise: Reference premise (article title + OCR text).

        Returns:
            Float in [0, 1].
        """
        claims = self.split_claims(generated)
        if not claims:
            return 0.0

        pipe = self._get_pipeline()
        n_entailed = 0
        for claim in claims:
            results = pipe(f"{premise}[SEP]{claim}")
            # results is a list of {label, score} dicts
            score_map = {r["label"].lower(): r["score"] for r in results[0]}
            entail = score_map.get("entailment", score_map.get("entail", 0.0))
            contra = score_map.get("contradiction", score_map.get("contradict", 0.0))
            if entail > contra:
                n_entailed += 1

        return n_entailed / len(claims)
