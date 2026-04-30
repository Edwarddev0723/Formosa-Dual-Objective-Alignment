"""formosa_dual.data.negative_sampler — in-batch negative tag sampling.

Supports three strategies:
- ``uniform``: draw M tags uniformly at random from the vocabulary.
- ``inverse_freq``: weight by 1/freq so rare tags are preferred.
- ``hard``: retrieve from a pre-computed embedding similarity cache.
"""
from __future__ import annotations

import random
from typing import Literal

import torch

from formosa_dual.data.tag_vocab import TagVocabulary
from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


class NegativeSampler:
    """Sample negative tag ids for contrastive learning.

    Args:
        vocab: Tag vocabulary.
        strategy: Sampling strategy.
        num_negatives: Number of negatives to sample per image.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        vocab: TagVocabulary,
        strategy: Literal["uniform", "inverse_freq", "hard"],
        num_negatives: int = 256,
        seed: int = 42,
    ) -> None:
        self._vocab = vocab
        self._strategy = strategy
        self._num_negatives = num_negatives
        self._rng = random.Random(seed)

        # Pre-compute sampling weights for non-uniform strategies
        self._weights: list[float] | None = None
        if strategy == "inverse_freq":
            self._weights = self._build_inverse_freq_weights(vocab)

        # Hard negative index (refreshed externally)
        self._hard_neg_index: dict[int, list[int]] | None = None

        logger.info(
            "NegativeSampler: strategy=%s num_negatives=%d vocab_size=%d",
            strategy,
            num_negatives,
            len(vocab),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_inverse_freq_weights(vocab: TagVocabulary) -> list[float]:
        """Return per-tag weights proportional to 1/freq (freq clipped to ≥1)."""
        weights: list[float] = []
        for tag in vocab.tags:
            tid = vocab.encode(tag)
            assert tid is not None
            freq = max(vocab.freq_of(tid), 1)
            weights.append(1.0 / freq)
        # Normalise (random.choices accepts unnormalised)
        return weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        positive_ids: list[int],
        visual_emb: torch.Tensor | None = None,
        tag_embs: torch.Tensor | None = None,
    ) -> list[int]:
        """Return a list of M negative tag ids, all different from *positive_ids*.

        Args:
            positive_ids: Tag ids that are positives for this image.
            visual_emb: Visual embedding ``[d]``; required when strategy=``hard``.
            tag_embs: Tag embeddings ``[K, d]``; required when strategy=``hard``.

        Returns:
            List of up to *num_negatives* negative tag ids.

        Raises:
            ValueError: If strategy=``hard`` and embeddings are not provided.
        """
        pos_set = set(positive_ids)
        vocab_size = len(self._vocab)
        all_ids = list(range(vocab_size))

        if self._strategy == "uniform":
            candidates = [tid for tid in all_ids if tid not in pos_set]
            k = min(self._num_negatives, len(candidates))
            return self._rng.sample(candidates, k)

        elif self._strategy == "inverse_freq":
            assert self._weights is not None
            candidates = [tid for tid in all_ids if tid not in pos_set]
            candidate_weights = [self._weights[tid] for tid in candidates]
            k = min(self._num_negatives, len(candidates))
            return self._rng.choices(candidates, weights=candidate_weights, k=k)

        elif self._strategy == "hard":
            if visual_emb is None or tag_embs is None:
                raise ValueError(
                    "strategy='hard' requires visual_emb and tag_embs; "
                    "call refresh_hard_neg_index before sampling, or supply embeddings."
                )
            # Cosine similarity between visual_emb and all tag embeddings
            v = visual_emb.float()
            t = tag_embs.float()
            if v.dim() == 1:
                v = v.unsqueeze(0)  # [1, d]
            sims = torch.nn.functional.cosine_similarity(v, t)  # [K]
            # Zero out positives
            for pid in pos_set:
                sims[pid] = -1.0
            # Top-M hardest negatives
            k = min(self._num_negatives, vocab_size - len(pos_set))
            topk = torch.topk(sims, k=k).indices.tolist()
            return topk

        else:
            raise ValueError(f"Unknown strategy: {self._strategy!r}")

    def refresh_hard_neg_index(self, model, dataloader) -> None:
        """Recompute the hard-negative cache from current embeddings.

        Called by the trainer every ``hard_neg_refresh_every_steps`` steps
        when strategy=``hard``.

        Args:
            model: :class:`~formosa_dual.models.dual_model.DualObjectiveModel`.
            dataloader: Training dataloader.
        """
        # ASSUMPTION: hard neg refresh reads model embeddings offline over the
        # dataloader and stores the top-K most similar tags per image in
        # self._hard_neg_index.  The model is expected to be in eval mode
        # during this call; the trainer is responsible for mode switching.
        logger.info("NegativeSampler: refreshing hard-negative index …")
        self._hard_neg_index = {}  # populated implementation deferred to trainer
        raise NotImplementedError(
            "refresh_hard_neg_index full implementation is driven by "
            "DualTrainer._refresh_hard_negatives().  This stub is intentional."
        )
