"""formosa_dual.losses.lambda_schedule — λ schedule for dual-objective weighting.

Controls how much weight the contrastive loss receives during training (§5.17).
"""
from __future__ import annotations

import math
from typing import Literal


class LambdaSchedule:
    """Schedule for the contrastive loss coefficient λ.

    Schedules:
    - ``constant``:       λ = peak
    - ``warmup``:         λ = peak * min(1, step / warmup_steps)
    - ``warmup_anneal``:  warmup to peak, then cosine decay to floor in the
                          last ``anneal_ratio`` fraction of training.

    Args:
        schedule: One of ``{"constant", "warmup", "warmup_anneal"}``.
        peak: Maximum λ value.
        floor: Minimum λ value (used only by ``warmup_anneal``).
        warmup_steps: Steps to reach ``peak`` (0 = immediate peak).
        total_steps: Total training steps (required for ``warmup_anneal``).
        anneal_ratio: Fraction of total_steps devoted to cosine annealing
                      (e.g. 0.3 means the last 30% of training).
    """

    def __init__(
        self,
        schedule: Literal["constant", "warmup", "warmup_anneal"],
        peak: float,
        floor: float = 0.0,
        warmup_steps: int = 0,
        total_steps: int = 1,
        anneal_ratio: float = 0.0,
    ) -> None:
        if schedule not in ("constant", "warmup", "warmup_anneal"):
            raise ValueError(f"Unknown schedule: {schedule!r}")
        self.schedule = schedule
        self.peak = peak
        self.floor = floor
        self.warmup_steps = max(warmup_steps, 0)
        self.total_steps = max(total_steps, 1)
        self.anneal_ratio = anneal_ratio

    def __call__(self, step: int) -> float:
        """Return λ at training *step* (0-indexed).

        Args:
            step: Current global training step.

        Returns:
            λ value as a Python float.
        """
        if self.schedule == "constant":
            return float(self.peak)

        if self.schedule == "warmup":
            if self.warmup_steps == 0:
                return float(self.peak)
            return float(self.peak * min(1.0, step / self.warmup_steps))

        # warmup_anneal
        anneal_start = int(self.total_steps * (1.0 - self.anneal_ratio))

        if step < self.warmup_steps:
            # Warmup phase
            if self.warmup_steps == 0:
                lam = self.peak
            else:
                lam = self.peak * (step / self.warmup_steps)
        elif step < anneal_start:
            # Plateau phase
            lam = self.peak
        else:
            # Cosine annealing phase
            anneal_steps = max(self.total_steps - anneal_start, 1)
            t = (step - anneal_start) / anneal_steps  # in [0, 1]
            lam = self.floor + 0.5 * (self.peak - self.floor) * (1.0 + math.cos(math.pi * t))

        return float(lam)
