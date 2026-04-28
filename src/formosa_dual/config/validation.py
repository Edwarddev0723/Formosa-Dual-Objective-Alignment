"""formosa_dual.config.validation — device-aware config validation and fallbacks.

Mutates the config in place with device-aware adjustments and logs any
changes as warnings so the user has a clear audit trail.
"""
from __future__ import annotations

import torch

from formosa_dual.config.schema import RunConfig
from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


def validate_config_for_device(cfg: RunConfig, device: torch.device) -> RunConfig:
    """Apply device-aware config fallbacks and validate compatibility.

    Mutates *cfg* in place.

    Args:
        cfg: Resolved :class:`~formosa_dual.config.schema.RunConfig`.
        device: Selected :class:`torch.device`.

    Returns:
        The (possibly mutated) ``cfg``.

    Raises:
        :class:`~formosa_dual.config.ConfigError`: On irreconcilable conflicts.
    """
    from formosa_dual.config import ConfigError

    if device.type == "mps":
        if cfg.optim.optimizer == "adamw_8bit":
            raise ConfigError(
                "8-bit optimizer requires bitsandbytes; not supported on MPS. "
                "Set optim.optimizer=adamw or use prod_gb10 profile."
            )
        if cfg.model.attn_implementation == "flash_attention_2":
            logger.warning(
                "flash_attention_2 is unavailable on MPS; falling back to sdpa"
            )
            cfg.model.attn_implementation = "sdpa"

    if device.type == "cpu":
        if cfg.device.mixed_precision != "no":
            logger.warning(
                "CPU does not support bf16/fp16 reliably; forcing mixed_precision=no"
            )
            cfg.device.mixed_precision = "no"

    return cfg
