"""formosa_dual.models.projection_head — two-layer MLP projection with L2 normalisation.

Projects a high-dimensional visual representation to the shared contrastive
embedding space (§5.12).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Two-layer MLP: Linear → GELU → Linear → L2-normalise.

    Default: ``3584 → 1024 → 256``.

    Args:
        d_in: Input dimension.
        d_hidden: Hidden layer dimension.
        d_out: Output embedding dimension.

    Input:
        ``[B, d_in]``

    Output:
        ``[B, d_out]``, unit-norm on last dimension.
    """

    def __init__(self, d_in: int, d_hidden: int = 1024, d_out: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalise.

        Args:
            x: ``[B, d_in]``

        Returns:
            ``[B, d_out]``, L2-normalised.
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=-1)
        return x
