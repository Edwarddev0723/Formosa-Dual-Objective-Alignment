"""Unit tests for ProjectionHead (§8.4)."""
import torch
import torch.nn.functional as F
import pytest
from formosa_dual.models.projection_head import ProjectionHead


def test_output_shape():
    head = ProjectionHead(d_in=128, d_hidden=64, d_out=32)
    x = torch.randn(4, 128)
    out = head(x)
    assert out.shape == (4, 32)


def test_output_is_unit_norm():
    head = ProjectionHead(d_in=128, d_hidden=64, d_out=32)
    x = torch.randn(8, 128)
    out = head(x)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_no_nan_in_output():
    head = ProjectionHead(d_in=256, d_hidden=128, d_out=64)
    x = torch.randn(4, 256)
    assert not torch.isnan(head(x)).any()


def test_gradient_flows():
    head = ProjectionHead(d_in=64, d_hidden=32, d_out=16)
    x = torch.randn(2, 64, requires_grad=True)
    out = head(x)
    out.sum().backward()
    assert x.grad is not None


def test_default_dimensions():
    head = ProjectionHead(d_in=3584)
    x = torch.randn(2, 3584)
    out = head(x)
    assert out.shape == (2, 256)
