"""Unit tests for checkpoint save/load roundtrip (§8.6)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from formosa_dual.training.checkpoint import load_checkpoint, save_checkpoint


# ---------------------------------------------------------------------------
# Stub model / optimizer / scheduler / accelerator
# ---------------------------------------------------------------------------


class _AuxModule(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)


class _StubModel(nn.Module):
    """Minimal stub that mimics DualObjectiveModel interface for checkpoint."""

    def __init__(self, d=16):
        super().__init__()
        self.backbone = nn.Linear(d, d)  # no save_pretrained → PEFT branch skipped
        self.pooler = _AuxModule(d)
        self.proj_head = _AuxModule(d)
        self.tag_projector = MagicMock()
        self.tag_projector.projector = _AuxModule(d)

    def get_trainable_param_groups(self):
        return []


def _stub_cfg(tmp_path):
    import types
    return types.SimpleNamespace(
        logging=types.SimpleNamespace(output_dir=str(tmp_path), run_name="ckpt_test"),
        model=MagicMock(),
        training=MagicMock(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_save_creates_expected_files(tmp_path):
    model = _StubModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    cfg = _stub_cfg(tmp_path)
    accelerator = MagicMock()

    ckpt_dir = save_checkpoint(
        model=model,
        optimizer=opt,
        scheduler=None,
        accelerator=accelerator,
        cfg=cfg,
        step=10,
        epoch=0,
        best_metric=None,
        output_dir=tmp_path,
    )

    assert ckpt_dir.is_dir()
    assert (ckpt_dir / "optimizer.pt").exists()
    assert (ckpt_dir / "rng_state.pt").exists()
    assert (ckpt_dir / "training_state.json").exists()
    # Either safetensors or .pt for aux
    assert (ckpt_dir / "aux_modules.safetensors").exists() or (ckpt_dir / "aux_modules.pt").exists()


def test_roundtrip_restores_aux_state(tmp_path):
    model = _StubModel(d=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    cfg = _stub_cfg(tmp_path)
    accelerator = MagicMock()

    # Mutate pooler weights
    with torch.no_grad():
        model.pooler.fc.weight.fill_(3.14)

    save_checkpoint(
        model=model, optimizer=opt, scheduler=None,
        accelerator=accelerator, cfg=cfg,
        step=5, epoch=0, best_metric=None, output_dir=tmp_path,
    )

    # Reset pooler
    with torch.no_grad():
        model.pooler.fc.weight.fill_(0.0)

    ckpt_dir = tmp_path / "checkpoint-5"
    state = load_checkpoint(
        model=model, optimizer=opt, scheduler=None,
        checkpoint_dir=ckpt_dir, accelerator=accelerator,
    )

    assert torch.allclose(model.pooler.fc.weight, torch.full((8, 8), 3.14), atol=1e-3)
    assert state["step"] == 5


def test_roundtrip_restores_optimizer_state(tmp_path):
    model = _StubModel(d=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Step once to populate optimizer state
    dummy_loss = model.backbone.weight.sum()
    dummy_loss.backward()
    opt.step()
    opt.zero_grad()

    cfg = _stub_cfg(tmp_path)
    accelerator = MagicMock()

    save_checkpoint(
        model=model, optimizer=opt, scheduler=None,
        accelerator=accelerator, cfg=cfg,
        step=1, epoch=0, best_metric=None, output_dir=tmp_path,
    )

    # New optimizer
    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-4)
    state = load_checkpoint(
        model=model, optimizer=opt2, scheduler=None,
        checkpoint_dir=tmp_path / "checkpoint-1", accelerator=accelerator,
    )
    assert state["step"] == 1
    # Check optimizer state was loaded
    assert "state" in opt2.state_dict()
