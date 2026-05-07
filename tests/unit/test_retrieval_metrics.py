"""Unit tests for image↔tag retrieval metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from formosa_dual.eval.retrieval_metrics import (
    compute_retrieval_metrics,
    evaluate_retrieval_loader,
)


def test_retrieval_metrics_perfect_alignment() -> None:
    """Perfect image/tag alignment gives perfect R@1 and mAP."""
    tag_embs = F.normalize(torch.eye(4), p=2, dim=-1)
    visual_embs = tag_embs[:3].clone()
    pos_tag_ids = torch.tensor([[0, -1], [1, -1], [2, -1]])
    pos_tag_mask = torch.tensor([[True, False], [True, False], [True, False]])

    metrics = compute_retrieval_metrics(
        visual_embs=visual_embs,
        tag_embs=tag_embs,
        pos_tag_ids=pos_tag_ids,
        pos_tag_mask=pos_tag_mask,
    )

    assert metrics["image_to_tag_R@1"] == 1.0
    assert metrics["image_to_tag_R@5"] == 1.0
    assert metrics["tag_to_image_R@1"] == 1.0
    assert metrics["tag_to_image_R@5"] == 1.0
    assert metrics["image_to_tag_mAP"] == 1.0
    assert metrics["tag_to_image_mAP"] == 1.0


def test_retrieval_metrics_multi_positive_hit() -> None:
    """An image query is a hit if any positive tag appears in top-k."""
    tag_embs = F.normalize(torch.eye(4), p=2, dim=-1)
    visual_embs = tag_embs[1:2].clone()
    pos_tag_ids = torch.tensor([[0, 1]])
    pos_tag_mask = torch.tensor([[True, True]])

    metrics = compute_retrieval_metrics(
        visual_embs=visual_embs,
        tag_embs=tag_embs,
        pos_tag_ids=pos_tag_ids,
        pos_tag_mask=pos_tag_mask,
    )

    assert metrics["image_to_tag_R@1"] == 1.0
    assert metrics["image_to_tag_R@5"] == 1.0
    assert metrics["n_images"] == 1.0
    assert metrics["n_eval_tags"] == 2.0


def test_evaluate_retrieval_loader_uses_visual_only_path() -> None:
    """Retrieval evaluation should not require a full model forward."""

    class _Vocab:
        def __len__(self):
            return 4

    class _TagProjector:
        def get_tag_embeddings(self, tag_ids):
            return F.normalize(
                torch.eye(4)[tag_ids.cpu()].to(tag_ids.device), p=2, dim=-1
            )

    class _Model:
        def __init__(self):
            self.tag_projector = _TagProjector()
            self.visual_calls = 0

        def eval(self):
            return self

        def encode_visual_embeddings(self, batch):
            self.visual_calls += 1
            return F.normalize(torch.eye(4)[:2], p=2, dim=-1)

        def __call__(self, batch):
            raise AssertionError("full forward should not be called for retrieval eval")

    batch = {
        "pos_tag_ids": torch.tensor([[0, -1], [1, -1]]),
        "pos_tag_mask": torch.tensor([[True, False], [True, False]]),
    }
    model = _Model()

    metrics = evaluate_retrieval_loader(
        model=model,
        dataloader=[batch],
        vocab=_Vocab(),
        device=torch.device("cpu"),
        show_progress=False,
    )

    assert model.visual_calls == 1
    assert metrics["image_to_tag_R@1"] == 1.0
