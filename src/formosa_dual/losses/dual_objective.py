"""formosa_dual.losses.dual_objective — composes caption + contrastive losses.

Implements the dual-objective loss combining CaptionLoss and
MultiPositiveInfoNCE with a λ schedule (§5.18).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from formosa_dual.config.schema import RunConfig
from formosa_dual.losses.caption_loss import CaptionLoss
from formosa_dual.losses.lambda_schedule import LambdaSchedule
from formosa_dual.losses.multi_pos_infonce import MultiPositiveInfoNCE
from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


class DualObjectiveLoss(nn.Module):
    """Compose caption loss and contrastive loss with a λ schedule.

    Args:
        cfg: Full :class:`~formosa_dual.config.schema.RunConfig`.
        total_steps: Total training steps (used to build the λ schedule).

    Forward:
        model_output: dict returned by ``DualObjectiveModel.forward()``.
        batch: dict from the data collator (used for ``labels``).
        step: Current global training step (int).

    Returns:
        Dict:
        ``{
            "loss": Tensor (scalar, for backward),
            "loss_caption": Tensor | float 0,
            "loss_contrast": Tensor | float 0,
            "lambda": float,
        }``
    """

    def __init__(self, cfg: RunConfig, total_steps: int) -> None:
        super().__init__()
        self.cfg = cfg
        self._caption_enabled = cfg.caption.enabled
        self._contrast_enabled = cfg.contrastive.enabled

        # Caption loss
        if self._caption_enabled:
            self.caption_loss = CaptionLoss(label_smoothing=cfg.caption.label_smoothing)

        # Contrastive loss + lambda schedule
        if self._contrast_enabled:
            self.infonce = MultiPositiveInfoNCE(tau=cfg.contrastive.tau)
            warmup_steps = int(cfg.contrastive.lambda_warmup_ratio * total_steps)
            self.lambda_schedule = LambdaSchedule(
                schedule=cfg.contrastive.lambda_schedule,
                peak=cfg.contrastive.lambda_value,
                floor=cfg.contrastive.lambda_floor,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                anneal_ratio=cfg.contrastive.lambda_anneal_ratio,
            )

    def forward(
        self,
        model_output: dict,
        batch: dict,
        step: int,
    ) -> dict:
        """Compute composite loss.

        Args:
            model_output: Output dict from ``DualObjectiveModel.forward()``.
            batch: Collator batch dict (must contain ``labels``).
            step: Current global training step.

        Returns:
            Dict with keys: ``loss``, ``loss_caption``, ``loss_contrast``, ``lambda``.
        """
        loss_caption: torch.Tensor | float = 0.0
        loss_contrast: torch.Tensor | float = 0.0
        lam: float = 0.0

        if self._caption_enabled:
            # Use the LM loss already computed by the backbone (more numerically stable)
            lm_loss = model_output.get("lm_loss")
            if lm_loss is not None:
                loss_caption = lm_loss
            else:
                # Fallback: recompute from logits (e.g., in tests without backbone)
                loss_caption = self.caption_loss(model_output["lm_logits"], batch["labels"])

        if self._contrast_enabled:
            visual_emb = model_output.get("visual_emb")
            tag_pos_emb = model_output.get("tag_pos_emb")
            tag_neg_emb = model_output.get("tag_neg_emb")
            pos_tag_mask = model_output.get("pos_tag_mask")

            if visual_emb is not None and tag_pos_emb is not None and tag_neg_emb is not None and pos_tag_mask is not None:
                loss_contrast = self.infonce(visual_emb, tag_pos_emb, pos_tag_mask, tag_neg_emb)
            lam = self.lambda_schedule(step)

        # Combine
        if self._caption_enabled and self._contrast_enabled:
            total_loss = loss_caption + lam * loss_contrast
        elif self._caption_enabled:
            total_loss = loss_caption
        else:
            # contrast only; spec says lambda=1.0 in V2 but we use the schedule
            total_loss = lam * loss_contrast if isinstance(loss_contrast, torch.Tensor) else torch.tensor(0.0)

        return {
            "loss": total_loss,
            "loss_caption": loss_caption,
            "loss_contrast": loss_contrast,
            "lambda": lam,
        }
