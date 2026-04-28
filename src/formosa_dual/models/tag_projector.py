"""formosa_dual.models.tag_projector — Chinese-CLIP-initialised tag embedding projector.

Encodes all vocabulary tags offline using Chinese-CLIP's text encoder,
freezes the base embeddings, and trains only a small projector head (§5.13).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from formosa_dual.data.tag_vocab import TagVocabulary
from formosa_dual.models.projection_head import ProjectionHead
from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


class TagProjector(nn.Module):
    """Project tag ids to the shared contrastive embedding space.

    Init steps:
    1. Load Chinese-CLIP text encoder.
    2. Encode all vocab tags to ``[K, 512]``, L2-normalise.
    3. Register as a frozen buffer ``tag_base_embs``.
    4. Build a trainable ``ProjectionHead(512, 1024, proj_dim)``.
    5. Free the Chinese-CLIP model from memory.

    Only ``self.projector`` trains; the base embeddings are frozen.

    Args:
        vocab: Tag vocabulary.
        chinese_clip_model: HuggingFace model id for Chinese-CLIP.
        proj_dim: Output projection dimension.
        device: Target device for the projector.
    """

    def __init__(
        self,
        vocab: TagVocabulary,
        chinese_clip_model: str,
        proj_dim: int = 256,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self._vocab = vocab
        self._proj_dim = proj_dim
        target_device = device or torch.device("cpu")

        # Encode tags offline and register as frozen buffer
        base_embs = self._encode_tags_with_clip(vocab, chinese_clip_model, target_device)
        self.register_buffer("tag_base_embs", base_embs)  # [K, 512], frozen

        # Trainable projector
        clip_dim = base_embs.shape[1]
        self.projector = ProjectionHead(clip_dim, d_hidden=1024, d_out=proj_dim)
        self.projector.to(target_device)

        logger.info(
            "TagProjector: %d tags, clip_dim=%d, proj_dim=%d",
            len(vocab),
            clip_dim,
            proj_dim,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_tags_with_clip(
        vocab: TagVocabulary,
        model_name: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode all vocabulary tags with the Chinese-CLIP text encoder.

        Returns ``[K, d_clip]`` float32 L2-normalised tensor.
        Frees the model from memory after encoding.
        """
        logger.info("Encoding %d tags with Chinese-CLIP (%s) …", len(vocab), model_name)

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers is required for Chinese-CLIP encoding") from exc

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        clip_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        clip_model.eval()
        clip_model.to(device)

        tags = vocab.tags  # ordered by id
        embs: list[torch.Tensor] = []

        batch_size = 64
        with torch.no_grad():
            for start in range(0, len(tags), batch_size):
                batch_tags = tags[start: start + batch_size]
                inputs = tokenizer(batch_tags, return_tensors="pt", padding=True, truncation=True, max_length=64)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Chinese-CLIP exposes text features via get_text_features or text_model.
                # Newer transformers may return None pooled_output, so fall back to the
                # text encoder's CLS token in that case.
                feat = None
                if hasattr(clip_model, "get_text_features"):
                    try:
                        feat = clip_model.get_text_features(**inputs)  # [B, d]
                    except TypeError:
                        feat = None
                if feat is None:
                    out = clip_model.text_model(**inputs)
                    feat = out.last_hidden_state[:, 0, :]  # CLS token

                feat = F.normalize(feat.float(), p=2, dim=-1)
                embs.append(feat.cpu())

        all_embs = torch.cat(embs, dim=0)  # [K, d]

        # Free Chinese-CLIP from memory
        del clip_model
        del tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info("Chinese-CLIP tag encoding complete; base emb shape: %s", list(all_embs.shape))
        return all_embs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tag_embeddings(self, tag_ids: torch.Tensor) -> torch.Tensor:
        """Look up base embeddings, project, and L2-normalise.

        Args:
            tag_ids: ``[B, P]`` integer tag ids (may include -1 for padding).

        Returns:
            ``[B, P, proj_dim]`` L2-normalised projected embeddings.
            Rows corresponding to -1 padding are zero vectors.
        """
        # tag_base_embs: [K, clip_dim]
        K = self.tag_base_embs.shape[0]

        # Clamp padding ids (e.g. -1) to 0 temporarily; we zero them out later
        clamped_ids = tag_ids.clamp(min=0)  # [B, P]
        pad_mask = tag_ids < 0  # [B, P]

        flat_ids = clamped_ids.reshape(-1)  # [B*P]
        flat_embs = self.tag_base_embs[flat_ids]  # [B*P, clip_dim]

        # Project
        projected = self.projector(flat_embs)  # [B*P, proj_dim], already L2 normed inside ProjectionHead

        B, P = tag_ids.shape
        projected = projected.view(B, P, self._proj_dim)

        # Zero out padded positions
        projected = projected.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return projected

    def forward(self, tag_ids: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`get_tag_embeddings`."""
        return self.get_tag_embeddings(tag_ids)
