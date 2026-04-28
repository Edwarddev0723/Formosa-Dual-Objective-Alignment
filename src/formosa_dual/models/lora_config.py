"""formosa_dual.models.lora_config — build a PEFT LoraConfig from RunConfig.

The exact LoRA targets are fixed by the spec (§1):
    q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
on the LM portion only.
"""
from __future__ import annotations

from formosa_dual.config.schema import LoRAConfig


def build_lora_config(cfg: LoRAConfig):
    """Return a ``peft.LoraConfig`` constructed from *cfg*.

    Args:
        cfg: :class:`~formosa_dual.config.schema.LoRAConfig`.

    Returns:
        ``peft.LoraConfig`` instance.

    Raises:
        ImportError: If ``peft`` is not installed.
    """
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise ImportError(
            "peft is required for LoRA support. "
            "Install it with: pip install peft>=0.13.0"
        ) from exc

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules),
        bias=cfg.bias,
        inference_mode=False,
    )
