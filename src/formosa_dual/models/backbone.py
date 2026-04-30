"""formosa_dual.models.backbone — load and freeze the Qwen2.5-VL backbone.

Architectural invariants (spec §1, §2):
- ViT is frozen by default; ``unfreeze_vit_last_n`` is the only lever.
- Vision-LM merger is always frozen.
- LoRA is applied to the LM portion only.
"""
from __future__ import annotations

from formosa_dual.config.schema import ModelConfig
from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


def load_backbone(cfg: ModelConfig) -> tuple:
    """Load the Qwen2.5-VL backbone and its processor.

    Args:
        cfg: :class:`~formosa_dual.config.schema.ModelConfig`.

    Returns:
        ``(model, processor)`` — a ``PreTrainedModel`` and its
        ``ProcessorMixin``.
    """
    import torch
    from transformers import AutoProcessor

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[cfg.torch_dtype]

    logger.info("Loading backbone: %s (dtype=%s, attn=%s)", cfg.name, cfg.torch_dtype, cfg.attn_implementation)

    model_cls = resolve_vision_lm_model_class(cfg.name)
    model = _from_pretrained_compat(model_cls, cfg, torch_dtype)
    processor = AutoProcessor.from_pretrained(cfg.name, trust_remote_code=True)

    logger.info("Backbone loaded: %s", cfg.name)
    return model, processor


def resolve_vision_lm_model_class(model_name: str):
    """Resolve a Transformers vision-language generation class across versions."""
    import transformers

    if "qwen2.5-vl" in model_name.lower() or "qwen2_5_vl" in model_name.lower():
        cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
        if cls is not None:
            return cls

    # Transformers 5 exposes the image-text auto class; Transformers 4 also has
    # the older Vision2Seq auto class. Prefer the newer name when available.
    for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        cls = getattr(transformers, class_name, None)
        if cls is not None:
            return cls

    raise ImportError(
        "No compatible vision-language model class found in transformers. "
        "Expected Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText, "
        "or AutoModelForVision2Seq."
    )


def _from_pretrained_compat(model_cls, cfg: ModelConfig, torch_dtype):
    """Call from_pretrained with dtype kwargs compatible with Transformers 4/5."""
    base_kwargs = {
        "revision": cfg.revision,
        "attn_implementation": cfg.attn_implementation,
        "trust_remote_code": True,
    }
    try:
        return model_cls.from_pretrained(cfg.name, dtype=torch_dtype, **base_kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        logger.info("Transformers class does not accept dtype=; retrying with torch_dtype=.")
        return model_cls.from_pretrained(cfg.name, torch_dtype=torch_dtype, **base_kwargs)


def apply_freeze_policy(model, cfg: ModelConfig) -> None:
    """Apply ViT / merger freeze policy in-place.

    - If ``cfg.freeze_vit=True``: freeze all visual encoder parameters.
    - If ``cfg.freeze_merger=True``: freeze merger parameters.
    - If ``cfg.unfreeze_vit_last_n > 0``: unfreeze the last N ViT layers
      regardless of ``freeze_vit``.

    Args:
        model: The loaded backbone (``PreTrainedModel``).
        cfg: :class:`~formosa_dual.config.schema.ModelConfig`.
    """
    # Freeze ViT
    vit = _get_visual_encoder(model)
    if vit is not None and cfg.freeze_vit:
        for param in vit.parameters():
            param.requires_grad = False
        logger.info("ViT frozen")

    # Unfreeze last N ViT transformer layers
    if vit is not None and cfg.unfreeze_vit_last_n > 0:
        layers = _get_vit_layers(vit)
        unfreeze_layers = layers[-cfg.unfreeze_vit_last_n:]
        for layer in unfreeze_layers:
            for param in layer.parameters():
                param.requires_grad = True
        logger.info("Unfroze last %d ViT layers", cfg.unfreeze_vit_last_n)

    # Freeze merger
    merger = _get_merger(model)
    if merger is not None and cfg.freeze_merger:
        for param in merger.parameters():
            param.requires_grad = False
        logger.info("Vision-LM merger frozen (always)")

    # Log trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params after freeze policy: %d / %d (%.2f%%)", trainable, total, 100 * trainable / max(total, 1))


def _get_visual_encoder(model):
    """Return the visual encoder sub-module, or None if not found."""
    for attr in ("visual", "model.visual", "vision_model", "model.vision_model"):
        obj = model
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and obj is not model:
            return obj
    return None


def _get_merger(model):
    """Return the vision-LM merger sub-module, or None if not found."""
    vit = _get_visual_encoder(model)
    if vit is None:
        return None
    for attr in ("merger", "projector", "connector"):
        obj = getattr(vit, attr, None)
        if obj is not None:
            return obj
    # Try on model directly
    for attr in ("merger", "projector"):
        obj = getattr(model, attr, None)
        if obj is not None:
            return obj
    return None


def _get_vit_layers(vit) -> list:
    """Return the list of ViT transformer layers, or empty list if unresolvable."""
    for attr in ("blocks", "layers", "encoder.layers", "encoder.blocks"):
        obj = vit
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            try:
                return list(obj)
            except TypeError:
                pass
    return []
