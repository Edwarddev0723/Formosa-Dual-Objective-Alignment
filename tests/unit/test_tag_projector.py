"""Tests for TagProjector output compatibility helpers."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from formosa_dual.models.tag_projector import _pool_text_features


def test_pool_text_features_accepts_tensor():
    tensor = torch.randn(2, 5)
    assert torch.equal(_pool_text_features(tensor), tensor)


def test_pool_text_features_prefers_pooler_output():
    output = SimpleNamespace(
        pooler_output=torch.ones(2, 5),
        last_hidden_state=torch.zeros(2, 3, 5),
    )
    pooled = _pool_text_features(output)
    assert torch.equal(pooled, torch.ones(2, 5))


def test_pool_text_features_uses_cls_from_last_hidden_state():
    hidden = torch.randn(2, 3, 5)
    output = SimpleNamespace(last_hidden_state=hidden)
    pooled = _pool_text_features(output)
    assert torch.equal(pooled, hidden[:, 0, :])


def test_pool_text_features_raises_for_unknown_output():
    with pytest.raises(TypeError):
        _pool_text_features(object())
