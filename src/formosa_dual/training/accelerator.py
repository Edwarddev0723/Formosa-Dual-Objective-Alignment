"""formosa_dual.training.accelerator — thin Accelerator wrapper.

Constructs an ``accelerate.Accelerator`` that respects
``cfg.device.mixed_precision`` and ``cfg.training.gradient_accumulation_steps``.
"""
from __future__ import annotations

from formosa_dual.config.schema import RunConfig
from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


def build_accelerator(cfg: RunConfig):
    """Construct an ``accelerate.Accelerator`` from config.

    Args:
        cfg: Full :class:`~formosa_dual.config.schema.RunConfig`.

    Returns:
        ``accelerate.Accelerator`` instance.
    """
    from accelerate import Accelerator

    # ``backend`` is one of {"wandb", "tensorboard", "none"}; map to accelerate's log_with.
    log_with = cfg.logging.backend if cfg.logging.backend != "none" else None

    accelerator = Accelerator(
        mixed_precision=cfg.device.mixed_precision,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        log_with=log_with,
    )
    logger.info(
        "Accelerator: mixed_precision=%s, grad_accum=%d, device=%s",
        cfg.device.mixed_precision,
        cfg.training.gradient_accumulation_steps,
        accelerator.device,
    )
    return accelerator
