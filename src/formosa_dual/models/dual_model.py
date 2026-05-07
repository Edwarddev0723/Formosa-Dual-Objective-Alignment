"""formosa_dual.models.dual_model — DualObjectiveModel.

Orchestrates the Qwen2.5-VL backbone (+ LoRA) + auxiliary modules
(AttentionPooler, ProjectionHead, TagProjector).

Forward contract:
    Returns a plain dict, never a loss scalar:
    {
        "lm_logits":    [B, L, V] float,
        "lm_loss":      scalar Tensor (raw CE from HF model),
        "visual_emb":   [B, proj_dim] or None,
        "tag_pos_emb":  [B, P_max, proj_dim] or None,
        "tag_neg_emb":  [B, M, proj_dim] or None,
        "pos_tag_mask": [B, P_max] bool or None,
    }

If ``cfg.contrastive.enabled`` is False, all visual/tag fields are None
and the pooler / proj_head / tag_projector are NOT instantiated.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from formosa_dual.config.schema import RunConfig
from formosa_dual.data.tag_vocab import TagVocabulary
from formosa_dual.models.attention_pooler import AttentionPooler
from formosa_dual.models.backbone import apply_freeze_policy, load_backbone
from formosa_dual.models.lora_config import build_lora_config
from formosa_dual.models.projection_head import ProjectionHead
from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


class DualObjectiveModel(nn.Module):
    """Backbone + LoRA + optional contrastive auxiliary modules.

    Args:
        cfg: Full :class:`~formosa_dual.config.schema.RunConfig`.
        vocab: Tag vocabulary.
        processor: Qwen2.5-VL processor (pre-loaded externally).

    Note:
        ``TagProjector`` is instantiated here; it calls into Chinese-CLIP
        to pre-encode all tag embeddings, then frees the CLIP model.
    """

    def __init__(
        self,
        cfg: RunConfig,
        vocab: TagVocabulary,
        processor: Any,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.processor = processor

        # ----------------------------------------------------------------
        # Backbone + LoRA
        # ----------------------------------------------------------------
        backbone, _ = load_backbone(cfg.model)
        apply_freeze_policy(backbone, cfg.model)

        if cfg.lora.enabled:
            try:
                from peft import get_peft_model
            except ImportError as exc:
                raise ImportError("peft is required; install peft>=0.13.0") from exc
            lora_cfg = build_lora_config(cfg.lora)
            backbone = get_peft_model(backbone, lora_cfg)
            backbone.print_trainable_parameters()
            logger.info("LoRA applied to backbone")

        if getattr(getattr(cfg, "training", None), "gradient_checkpointing", False):
            self._enable_gradient_checkpointing(backbone)

        self.backbone = backbone

        # Resolve LM hidden size from the backbone config (no schema field needed).
        # Qwen2.5-VL exposes either ``config.hidden_size`` directly or
        # ``config.text_config.hidden_size`` depending on transformers version.
        d_lm = self._resolve_hidden_size(backbone)

        # ----------------------------------------------------------------
        # Forward hook to capture visual merger output
        # ----------------------------------------------------------------
        self._cached_visual_tokens: torch.Tensor | None = None
        self._hook_handle = self._register_merger_hook()

        # ----------------------------------------------------------------
        # Contrastive aux modules (only when enabled)
        # ----------------------------------------------------------------
        self.pooler: AttentionPooler | None = None
        self.proj_head: ProjectionHead | None = None
        self.tag_projector = None  # late import to avoid circular

        if cfg.contrastive.enabled:
            proj_dim = cfg.aux.proj_dim

            self.pooler = AttentionPooler(d_lm=d_lm, num_heads=cfg.aux.pooler_num_heads)
            self.proj_head = ProjectionHead(
                d_in=d_lm, d_hidden=cfg.aux.proj_hidden, d_out=proj_dim
            )

            from formosa_dual.models.tag_projector import TagProjector

            # TagProjector will encode all tags; device determined later when .to(device) is called
            self.tag_projector = TagProjector(
                vocab=vocab,
                chinese_clip_model=cfg.aux.chinese_clip_model,
                proj_dim=proj_dim,
                device=None,
            )
            logger.info("Contrastive aux modules initialised (proj_dim=%d)", proj_dim)
        else:
            logger.info("Contrastive disabled; aux modules not instantiated")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_hidden_size(backbone) -> int:
        """Return the LM hidden size from the backbone config.

        Qwen2.5-VL-7B-Instruct uses ``config.hidden_size = 3584``; the 3B
        variant uses 2048. PEFT-wrapped models still expose ``.config``.
        """
        cfg = getattr(backbone, "config", None)
        if cfg is None and hasattr(backbone, "base_model"):
            cfg = getattr(backbone.base_model, "config", None)
        if cfg is None:
            raise RuntimeError("Cannot resolve hidden_size: backbone has no .config")
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
            return int(text_cfg.hidden_size)
        if hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
        raise RuntimeError("Cannot resolve hidden_size from backbone.config")

    @staticmethod
    def _enable_gradient_checkpointing(backbone) -> None:
        """Enable HF gradient checkpointing and disable KV cache when available."""
        for config_obj in (
            getattr(backbone, "config", None),
            getattr(getattr(backbone, "base_model", None), "config", None),
        ):
            if config_obj is not None and hasattr(config_obj, "use_cache"):
                config_obj.use_cache = False

        if hasattr(backbone, "gradient_checkpointing_enable"):
            try:
                backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                backbone.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled on backbone")
        else:
            logger.warning("Backbone does not expose gradient_checkpointing_enable()")

        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
            logger.info("Input gradients enabled for checkpointed LoRA training")

    # ------------------------------------------------------------------
    # Forward hook registration
    # ------------------------------------------------------------------

    def _register_merger_hook(self):
        """Register a hook on the vision-LM merger to capture visual tokens.

        Attempts the following attribute paths in order:
        1. backbone.model.visual.merger   (Qwen2.5-VL standard)
        2. backbone.visual.merger
        3. backbone.model.visual
        Logs a warning if the merger cannot be found; visual_emb will be None.
        """
        # When LoRA wraps the model, the actual backbone is under .base_model.model
        root = self.backbone
        candidates = [
            "model.visual.merger",
            "visual.merger",
            "base_model.model.model.visual.merger",
            "base_model.model.visual.merger",
        ]
        for path in candidates:
            obj = root
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                logger.info("Registering visual merger hook at path: %s", path)
                handle = obj.register_forward_hook(self._merger_hook)
                return handle

        logger.warning(
            "Could not find visual merger submodule — contrastive visual_emb will be None. "
            "Attempted paths: %s",
            candidates,
        )
        return None

    def _merger_hook(self, module, input, output):  # noqa: ARG002
        """Forward hook: cache merger output tensor."""
        # output shape: [B, N_v, d_lm] or similar
        self._cached_visual_tokens = (
            output if isinstance(output, torch.Tensor) else output[0]
        )

    def encode_visual_embeddings(self, batch: dict) -> torch.Tensor | None:
        """Encode only the visual branch for retrieval-style evaluation.

        This bypasses the LM logits/loss computation used by ``forward`` and
        therefore is much faster for retrieval metrics. If the backbone wrapper
        does not expose a callable visual module, it falls back to ``forward``.
        """
        if not self.cfg.contrastive.enabled:
            return None

        visual_tokens = self._extract_visual_tokens_direct(batch)
        if visual_tokens is None:
            return self.forward(batch).get("visual_emb")
        return self._visual_tokens_to_embedding(batch, visual_tokens)

    def _extract_visual_tokens_direct(self, batch: dict) -> torch.Tensor | None:
        """Call the Qwen visual tower directly when its module is reachable."""
        visual = self._get_visual_module()
        pixel_values = batch.get("pixel_values")
        if visual is None or pixel_values is None or not callable(visual):
            return None

        pixel_values = self._cast_pixel_values_for_visual(pixel_values, visual)
        image_grid_thw = batch.get("image_grid_thw")
        try:
            if image_grid_thw is not None:
                return visual(pixel_values, grid_thw=image_grid_thw)
            return visual(pixel_values)
        except TypeError:
            if image_grid_thw is None:
                return None
        try:
            return visual(pixel_values, image_grid_thw=image_grid_thw)
        except TypeError:
            return None

    def _get_visual_module(self):
        """Return the callable visual encoder module under common PEFT/Qwen paths."""
        candidates = [
            "model.visual",
            "visual",
            "base_model.model.model.visual",
            "base_model.model.visual",
        ]
        for path in candidates:
            obj = self.backbone
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        return None

    @staticmethod
    def _cast_pixel_values_for_visual(
        pixel_values: torch.Tensor, visual
    ) -> torch.Tensor:
        """Match pixel dtype to the visual module dtype when possible."""
        if not torch.is_floating_point(pixel_values):
            return pixel_values
        target_dtype = getattr(visual, "dtype", None)
        if target_dtype is None and hasattr(visual, "parameters"):
            first_param = next(visual.parameters(), None)
            target_dtype = first_param.dtype if first_param is not None else None
        if target_dtype is not None and pixel_values.dtype != target_dtype:
            return pixel_values.to(dtype=target_dtype)
        return pixel_values

    def _visual_tokens_to_embedding(
        self,
        batch: dict,
        visual_tokens: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | None:
        """Reshape visual tokens per image, then pool and project them."""
        if self.pooler is None or self.proj_head is None:
            return None

        if not isinstance(visual_tokens, torch.Tensor):
            visual_tokens = visual_tokens[0]

        B = (
            batch["input_ids"].size(0)
            if "input_ids" in batch
            else batch["pos_tag_ids"].size(0)
        )
        if visual_tokens.dim() == 2:
            if B == 1:
                visual_tokens = visual_tokens.unsqueeze(0)
            else:
                thw = batch.get("image_grid_thw")
                if thw is None:
                    raise RuntimeError(
                        "image_grid_thw missing but B>1 with 2-D visual tokens; "
                        "cannot reshape per-image."
                    )
                merge = getattr(
                    getattr(
                        self.backbone.config, "vision_config", self.backbone.config
                    ),
                    "spatial_merge_size",
                    2,
                )
                per_img = (
                    thw[:, 0] * thw[:, 1] * thw[:, 2] // (merge * merge)
                ).tolist()
                chunks = list(torch.split(visual_tokens, per_img, dim=0))
                max_n = max(c.size(0) for c in chunks)
                padded = torch.zeros(
                    B,
                    max_n,
                    visual_tokens.size(-1),
                    dtype=visual_tokens.dtype,
                    device=visual_tokens.device,
                )
                for i, c in enumerate(chunks):
                    padded[i, : c.size(0)] = c
                visual_tokens = padded

        pooler_dtype = next(self.pooler.parameters()).dtype
        if visual_tokens.dtype != pooler_dtype:
            visual_tokens = visual_tokens.to(dtype=pooler_dtype)
        pooled = self.pooler(visual_tokens)
        return self.proj_head(pooled)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        """Run backbone + optional contrastive pipeline.

        Args:
            batch: Dict from :class:`~formosa_dual.data.collator.DualCollator`:
                ``input_ids``, ``attention_mask``, ``labels``,
                ``pixel_values``, ``image_grid_thw`` (optional),
                ``pos_tag_ids``, ``pos_tag_mask``,
                ``neg_tag_ids``.

        Returns:
            Dict with keys: ``lm_logits``, ``lm_loss``,
            ``visual_emb``, ``tag_pos_emb``, ``tag_neg_emb``, ``pos_tag_mask``.
        """
        # Reset cached visual tokens before each forward
        self._cached_visual_tokens = None

        # ----------------------------------------------------------------
        # Backbone forward
        # ----------------------------------------------------------------
        backbone_kwargs: dict[str, Any] = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
            "pixel_values": batch["pixel_values"],
        }
        if "image_grid_thw" in batch and batch["image_grid_thw"] is not None:
            backbone_kwargs["image_grid_thw"] = batch["image_grid_thw"]

        backbone_out = self.backbone(**backbone_kwargs)

        lm_logits = backbone_out.logits  # [B, L, V]
        lm_loss = backbone_out.loss  # scalar

        # ----------------------------------------------------------------
        # Contrastive branch
        # ----------------------------------------------------------------
        visual_emb: torch.Tensor | None = None
        tag_pos_emb: torch.Tensor | None = None
        tag_neg_emb: torch.Tensor | None = None
        pos_tag_mask: torch.Tensor | None = None

        if self.cfg.contrastive.enabled and self._cached_visual_tokens is not None:
            # Qwen2.5-VL's `visual.merger` returns a 2-D tensor of shape
            # ``[total_visual_tokens, d_lm]`` — flattened over the batch.
            # Reshape to ``[B, N_v, d_lm]`` (padded) before pooling.
            visual_emb = self._visual_tokens_to_embedding(
                batch, self._cached_visual_tokens
            )

            # Tag embeddings
            pos_ids = batch["pos_tag_ids"]  # [B, P_max]
            neg_ids = batch["neg_tag_ids"]  # [B, M]
            pos_tag_mask = batch["pos_tag_mask"]  # [B, P_max] bool

            tag_pos_emb = self.tag_projector(pos_ids)  # [B, P_max, proj_dim]
            tag_neg_emb = self.tag_projector(neg_ids)  # [B, M, proj_dim]

        return {
            "lm_logits": lm_logits,
            "lm_loss": lm_loss,
            "visual_emb": visual_emb,
            "tag_pos_emb": tag_pos_emb,
            "tag_neg_emb": tag_neg_emb,
            "pos_tag_mask": pos_tag_mask,
        }

    # ------------------------------------------------------------------
    # Param groups
    # ------------------------------------------------------------------

    def get_trainable_param_groups(self) -> list[dict]:
        """Return optimizer parameter groups.

        Groups:
        1. ``"lora"``     — LoRA adapter parameters.
        2. ``"aux"``      — AttentionPooler + ProjectionHead.
        3. ``"tag_proj"`` — TagProjector.projector only.

        Groups with no trainable parameters are omitted.
        """
        opt = self.cfg.optim
        groups: list[dict] = []

        # LoRA params
        lora_params = [
            p
            for n, p in self.backbone.named_parameters()
            if p.requires_grad and "lora" in n.lower()
        ]
        if lora_params:
            groups.append(
                {
                    "params": lora_params,
                    "lr": opt.lr_lora,
                    "weight_decay": opt.weight_decay_lora,
                    "name": "lora",
                }
            )

        # Aux params (pooler + proj_head)
        aux_params: list[torch.nn.Parameter] = []
        if self.pooler is not None:
            aux_params.extend(p for p in self.pooler.parameters() if p.requires_grad)
        if self.proj_head is not None:
            aux_params.extend(p for p in self.proj_head.parameters() if p.requires_grad)
        if aux_params:
            groups.append(
                {
                    "params": aux_params,
                    "lr": opt.lr_aux,
                    "weight_decay": opt.weight_decay_aux,
                    "name": "aux",
                }
            )

        # Tag projector (projector head only, not the frozen buffer)
        if self.tag_projector is not None:
            tag_params = [
                p for p in self.tag_projector.projector.parameters() if p.requires_grad
            ]
            if tag_params:
                groups.append(
                    {
                        "params": tag_params,
                        "lr": opt.lr_aux,
                        "weight_decay": opt.weight_decay_tag_proj,
                        "name": "tag_proj",
                    }
                )

        return groups
