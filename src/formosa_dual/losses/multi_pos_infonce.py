"""formosa_dual.losses.multi_pos_infonce — multi-positive InfoNCE loss.

Reference implementation provided verbatim from spec §5.16.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiPositiveInfoNCE(nn.Module):
    """Multi-positive InfoNCE contrastive loss.

    For sample i with positives P_i and negatives N_i^-:

        L_i = -1/|P_i| sum_{p in P_i} log [
            exp(<v_i, t_p>/τ) /
            sum_{k in P_i ∪ N_i^-} exp(<v_i, t_k>/τ)
        ]

    where <,> is cosine similarity (inputs assumed L2-normalised).

    Args:
        tau: Temperature (default 0.07).

    Forward:
        v:        ``[B, d]`` visual embeddings (L2-normalised)
        pos_t:    ``[B, P_max, d]`` positive tag embeddings (L2-normalised)
        pos_mask: ``[B, P_max]`` bool (``True`` = valid positive)
        neg_t:    ``[B, M, d]`` negative tag embeddings (L2-normalised)

    Returns:
        Scalar mean loss.
    """

    def __init__(self, tau: float = 0.07) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.tau = tau

    def forward(
        self,
        v: torch.Tensor,
        pos_t: torch.Tensor,
        pos_mask: torch.Tensor,
        neg_t: torch.Tensor,
    ) -> torch.Tensor:
        # Shape checks
        assert v.dim() == 2
        assert pos_t.dim() == 3
        assert neg_t.dim() == 3
        assert pos_mask.dtype == torch.bool

        v_exp = v.unsqueeze(1)                                       # [B, 1, d]
        pos_sim = (v_exp * pos_t).sum(-1) / self.tau                  # [B, P]
        neg_sim = (v_exp * neg_t).sum(-1) / self.tau                  # [B, M]

        all_sim = torch.cat([pos_sim, neg_sim], dim=1)                # [B, P+M]
        pos_valid = pos_mask.float()
        neg_valid = torch.ones_like(neg_sim)
        all_valid = torch.cat([pos_valid, neg_valid], dim=1)          # [B, P+M]

        all_sim_masked = all_sim.masked_fill(all_valid == 0, float("-inf"))
        log_denom = torch.logsumexp(all_sim_masked, dim=1)            # [B]

        log_prob_per_pos = pos_sim - log_denom.unsqueeze(1)           # [B, P]
        log_prob_per_pos = log_prob_per_pos.masked_fill(~pos_mask, 0.0)

        n_pos = pos_mask.sum(dim=1).clamp(min=1).float()
        loss_per_sample = -(log_prob_per_pos.sum(dim=1) / n_pos)
        return loss_per_sample.mean()
