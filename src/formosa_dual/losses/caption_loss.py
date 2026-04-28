"""formosa_dual.losses.caption_loss — standard cross-entropy on shifted labels.

Wraps the LM loss already computed by the backbone with an optional
label-smoothing override.  In the common case ``label_smoothing=0.0``
this is a no-op wrapper that simply returns the raw LM loss scalar.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptionLoss(nn.Module):
    """Cross-entropy caption loss, optionally with label smoothing.

    Args:
        label_smoothing: Label-smoothing epsilon (default 0.0 = off).

    Forward:
        logits: ``[B, L, V]``
        labels: ``[B, L]`` — positions with -100 are ignored.

    Returns:
        Scalar mean loss.
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        if label_smoothing < 0.0 or label_smoothing >= 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute mean CE loss over non-ignored tokens.

        Args:
            logits: ``[B, L, V]``
            labels: ``[B, L]``

        Returns:
            Scalar mean cross-entropy.
        """
        # Shift: predict token at position t+1 from position t
        shift_logits = logits[:, :-1, :].contiguous()     # [B, L-1, V]
        shift_labels = labels[:, 1:].contiguous()          # [B, L-1]

        B, L, V = shift_logits.shape
        loss = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )
        return loss
