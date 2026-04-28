"""Unit tests for AttentionPooler (§8.4)."""
import torch
import pytest
from formosa_dual.models.attention_pooler import AttentionPooler


def test_output_shape():
    pooler = AttentionPooler(d_lm=64, num_heads=4)
    tokens = torch.randn(2, 16, 64)
    out = pooler(tokens)
    assert out.shape == (2, 64)


def test_output_shape_with_mask():
    pooler = AttentionPooler(d_lm=64, num_heads=4)
    tokens = torch.randn(3, 10, 64)
    mask = torch.ones(3, 10, dtype=torch.bool)
    mask[0, 5:] = False  # pad last 5 positions for sample 0
    out = pooler(tokens, mask)
    assert out.shape == (3, 64)


def test_no_nan_in_output():
    pooler = AttentionPooler(d_lm=128, num_heads=8)
    tokens = torch.randn(4, 20, 128)
    out = pooler(tokens)
    assert not torch.isnan(out).any()


def test_learnable_query_is_parameter():
    pooler = AttentionPooler(d_lm=64, num_heads=4)
    param_names = [n for n, _ in pooler.named_parameters()]
    assert "query" in param_names


def test_gradient_flows_through_pooler():
    pooler = AttentionPooler(d_lm=64, num_heads=4)
    tokens = torch.randn(2, 8, 64, requires_grad=True)
    out = pooler(tokens)
    loss = out.sum()
    loss.backward()
    assert tokens.grad is not None
