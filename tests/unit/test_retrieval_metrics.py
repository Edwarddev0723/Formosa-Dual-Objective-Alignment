"""Unit tests for image↔tag retrieval metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from formosa_dual.eval.retrieval_metrics import compute_retrieval_metrics


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
