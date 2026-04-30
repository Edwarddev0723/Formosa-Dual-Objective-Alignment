"""formosa_dual.data.collator — DualCollator that assembles model-ready batches.

Handles:
- Qwen2.5-VL processor invocation (chat template + image/text tokenization).
- Caption label construction with -100 mask on prompt tokens.
- Positive-tag padding to P_max with -1 sentinel and bool mask.
- Negative-tag sampling via :class:`~formosa_dual.data.negative_sampler.NegativeSampler`.
"""
from __future__ import annotations

import torch

from formosa_dual.data.negative_sampler import NegativeSampler
from formosa_dual.data.tag_vocab import TagVocabulary
from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)

_CAPTION_PROMPT_TEMPLATE = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "請描述這張圖片的文化內容。<|im_end|>\n<|im_start|>assistant\n"
)


class DualCollator:
    """Collate a list of :class:`~formosa_dual.data.dataset.FormosaDataset` items
    into a single model-ready batch dict.

    Output batch keys:
        ``pixel_values``, ``input_ids``, ``attention_mask``, ``labels``,
        ``image_grid_thw``, ``pos_tag_ids``, ``pos_tag_mask``, ``neg_tag_ids``.

    Args:
        processor: Qwen2.5-VL processor from ``AutoProcessor.from_pretrained``.
        vocab: Tag vocabulary.
        negative_sampler: Sampler for negative tag ids.
        max_caption_tokens: Maximum total sequence length (prompt + caption).
        max_pos_tags: Maximum number of positive tags per sample (pad/truncate).
    """

    def __init__(
        self,
        processor,
        vocab: TagVocabulary,
        negative_sampler: NegativeSampler,
        max_caption_tokens: int = 384,
        max_pos_tags: int = 10,
    ) -> None:
        self._processor = processor
        self._vocab = vocab
        self._neg_sampler = negative_sampler
        self._max_tokens = max_caption_tokens
        self._max_pos = max_pos_tags

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Collate a batch of dataset items.

        Args:
            batch: List of dicts from :meth:`FormosaDataset.__getitem__`.

        Returns:
            Batch dict with tensors ready for :class:`~formosa_dual.models.dual_model.DualObjectiveModel`.
        """
        images = [item["image"] for item in batch]
        captions = [item["caption"] for item in batch]
        pos_tag_ids_list = [item["pos_tag_ids"] for item in batch]

        # ------------------------------------------------------------------
        # 1. Build conversation messages for the Qwen processor
        # ------------------------------------------------------------------
        messages_batch = []
        for caption in captions:
            messages_batch.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "請描述這張圖片的文化內容。"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": caption}],
                },
            ])

        texts = [
            self._processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in messages_batch
        ]

        # ------------------------------------------------------------------
        # 2. Process images + text through the Qwen processor
        # ------------------------------------------------------------------
        encoding = self._processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self._max_tokens,
            return_tensors="pt",
        )

        input_ids: torch.Tensor = encoding["input_ids"]  # [B, L]
        attention_mask: torch.Tensor = encoding["attention_mask"]  # [B, L]

        # image_grid_thw is Qwen2.5-VL specific; may not always be present
        image_grid_thw = encoding.get("image_grid_thw", None)

        pixel_values = encoding.get("pixel_values", None)

        # ------------------------------------------------------------------
        # 3. Build labels: clone input_ids, mask prompt tokens with -100
        # ------------------------------------------------------------------
        labels = self._build_labels(input_ids, texts)

        # ------------------------------------------------------------------
        # 4. Pad positive tag ids to [B, max_pos_tags]
        # ------------------------------------------------------------------
        B = len(batch)
        pos_tag_ids_t = torch.full((B, self._max_pos), fill_value=-1, dtype=torch.long)
        pos_tag_mask = torch.zeros(B, self._max_pos, dtype=torch.bool)

        for i, ids in enumerate(pos_tag_ids_list):
            ids = ids[: self._max_pos]
            n = len(ids)
            if n > 0:
                pos_tag_ids_t[i, :n] = torch.tensor(ids, dtype=torch.long)
                pos_tag_mask[i, :n] = True

        # ------------------------------------------------------------------
        # 5. Sample negatives for each item
        # ------------------------------------------------------------------
        neg_ids_list: list[list[int]] = []
        for ids in pos_tag_ids_list:
            neg_ids_list.append(self._neg_sampler.sample(ids))

        M = max(len(n) for n in neg_ids_list) if neg_ids_list else 0
        neg_tag_ids_t = torch.full((B, M), fill_value=-1, dtype=torch.long)
        for i, ids in enumerate(neg_ids_list):
            if ids:
                neg_tag_ids_t[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        out: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pos_tag_ids": pos_tag_ids_t,
            "pos_tag_mask": pos_tag_mask,
            "neg_tag_ids": neg_tag_ids_t,
        }

        if pixel_values is not None:
            out["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            out["image_grid_thw"] = image_grid_thw

        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_labels(self, input_ids: torch.Tensor, texts: list[str]) -> torch.Tensor:
        """Build caption labels with -100 masking on non-assistant tokens.

        Strategy: locate the assistant turn start token and mask everything
        before it.  Falls back to cloning input_ids entirely if the turn
        boundary cannot be identified (all tokens become supervised).
        """
        labels = input_ids.clone()

        # Find where the assistant response begins in tokenized form.
        # We re-tokenize just the prompt portion and mask those positions.
        try:
            assistant_token_id = self._processor.tokenizer.convert_tokens_to_ids(
                "<|im_start|>"
            )
        except Exception:
            assistant_token_id = None

        if assistant_token_id is not None:
            for i, input_row in enumerate(input_ids):
                # Find the last occurrence of <|im_start|> (assistant turn)
                positions = (input_row == assistant_token_id).nonzero(as_tuple=True)[0]
                if len(positions) >= 2:
                    # Second <|im_start|> is the assistant turn
                    assistant_start = positions[-1].item()
                    labels[i, : assistant_start + 1] = -100
                else:
                    # Cannot identify prompt; mask nothing (all supervised)
                    pass
        else:
            # Fallback: mask nothing (treat full sequence as supervised)
            pass

        # Mask padding tokens
        labels[labels == self._processor.tokenizer.pad_token_id] = -100

        return labels
