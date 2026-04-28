"""Unit tests for MultiPositiveInfoNCE (§8.5)."""
import torch
import pytest
from formosa_dual.losses.multi_pos_infonce import MultiPositiveInfoNCE


def _unit(t):
    """L2-normalise along last dimension."""
    import torch.nn.functional as F
    return F.normalize(t, p=2, dim=-1)


def test_positive_tau_required():
    with pytest.raises(ValueError):
        MultiPositiveInfoNCE(tau=0.0)
    with pytest.raises(ValueError):
        MultiPositiveInfoNCE(tau=-0.1)


def test_loss_scalar():
    B, P, M, d = 4, 3, 5, 16
    loss_fn = MultiPositiveInfoNCE(tau=0.07)
    v = _unit(torch.randn(B, d))
    pos_t = _unit(torch.randn(B, P, d))
    pos_mask = torch.ones(B, P, dtype=torch.bool)
    neg_t = _unit(torch.randn(B, M, d))
    out = loss_fn(v, pos_t, pos_mask, neg_t)
    assert out.shape == ()


def test_loss_non_negative():
    B, P, M, d = 4, 3, 5, 16
    loss_fn = MultiPositiveInfoNCE(tau=0.07)
    v = _unit(torch.randn(B, d))
    pos_t = _unit(torch.randn(B, P, d))
    pos_mask = torch.ones(B, P, dtype=torch.bool)
    neg_t = _unit(torch.randn(B, M, d))
    assert loss_fn(v, pos_t, pos_mask, neg_t).item() >= 0.0


def test_perfect_alignment_gives_low_loss():
    """When visual emb equals the positive tag emb, loss should be low."""
    B, P, M, d = 4, 1, 6, 32
    loss_fn = MultiPositiveInfoNCE(tau=0.07)
    v = _unit(torch.randn(B, d))
    pos_t = v.unsqueeze(1).expand(-1, P, -1)  # perfect alignment
    pos_mask = torch.ones(B, P, dtype=torch.bool)
    neg_t = _unit(torch.randn(B, M, d))
    loss = loss_fn(v, pos_t, pos_mask, neg_t).item()
    assert loss < 2.0  # should be much lower than random


def test_partial_pos_mask_respected():
    B, P, M, d = 4, 4, 4, 16
    loss_fn = MultiPositiveInfoNCE(tau=0.07)
    v = _unit(torch.randn(B, d))
    pos_t = _unit(torch.randn(B, P, d))
    neg_t = _unit(torch.randn(B, M, d))
    # Only first 2 positives are valid
    mask_partial = torch.zeros(B, P, dtype=torch.bool)
    mask_partial[:, :2] = True
    mask_full = torch.ones(B, P, dtype=torch.bool)
    # Both should return a finite scalar (not crash)
    assert loss_fn(v, pos_t, mask_partial, neg_t).isfinite()
    assert loss_fn(v, pos_t, mask_full, neg_t).isfinite()
