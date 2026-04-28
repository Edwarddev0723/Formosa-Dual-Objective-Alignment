"""formosa_dual.models.attention_pooler — learnable attention pooling over visual tokens.

Pools variable-length visual token sequences to a single vector via a
learned query and multi-head attention (§5.11).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooler(nn.Module):
    """Pool visual token sequences to a single vector via a learned query.

    Uses :class:`torch.nn.MultiheadAttention` with ``batch_first=True``.

    Args:
        d_lm: Visual token / language model hidden dimension.
        num_heads: Number of attention heads.

    Input:
        visual_tokens: ``[B, N_v, d_lm]``
        attention_mask: ``[B, N_v]`` bool (``True`` = valid token)

    Output:
        ``[B, d_lm]``
    """

    def __init__(self, d_lm: int = 3584, num_heads: int = 8) -> None:
        super().__init__()
        self.d_lm = d_lm
        self.num_heads = num_heads

        self.query = nn.Parameter(torch.zeros(1, 1, d_lm))
        nn.init.normal_(self.query, std=0.02)

        self.attn = nn.MultiheadAttention(d_lm, num_heads, batch_first=True)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool visual tokens to a single vector.

        Args:
            visual_tokens: ``[B, N_v, d_lm]``
            attention_mask: ``[B, N_v]`` bool, ``True`` = valid.
                If ``None``, all positions are treated as valid.

        Returns:
            Pooled tensor ``[B, d_lm]``.
        """
        B = visual_tokens.size(0)
        query = self.query.expand(B, -1, -1)  # [B, 1, d_lm]

        # MHA expects key_padding_mask: True means *ignore*.
        # Our attention_mask: True = valid → invert for key_padding_mask.
        key_padding_mask: torch.Tensor | None = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask  # [B, N_v], True = pad

        attn_out, _ = self.attn(
            query=query,
            key=visual_tokens,
            value=visual_tokens,
            key_padding_mask=key_padding_mask,
        )  # attn_out: [B, 1, d_lm]

        return attn_out.squeeze(1)  # [B, d_lm]
