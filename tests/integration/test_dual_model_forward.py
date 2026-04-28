"""Unit tests for DualObjectiveModel forward — uses a lightweight stub backbone
to avoid downloading the full Qwen2.5-VL checkpoint (§8.4).

Tests verify:
- forward returns expected keys
- visual_emb is None when contrastive disabled
- get_trainable_param_groups returns named groups
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers: stub backbone + config
# ---------------------------------------------------------------------------


class _FakeOutput:
    def __init__(self, B, L, V):
        self.logits = torch.randn(B, L, V)
        self.loss = torch.tensor(1.5)


class _FakeBackbone(nn.Module):
    """Minimal backbone stub that returns HF-style output."""

    def __init__(self, d_lm=64, V=1000):
        super().__init__()
        self.d_lm = d_lm
        self.V = V
        # Expose config.hidden_size so DualObjectiveModel._resolve_hidden_size() works.
        self.config = types.SimpleNamespace(hidden_size=d_lm, text_config=None)
        # Minimal LoRA-ish parameter for param-group detection
        self.lora_A = nn.Parameter(torch.randn(4, d_lm))
        # Merger module: its forward will be called by our stub to trigger the hook
        self.visual = types.SimpleNamespace(
            merger=nn.Linear(d_lm, d_lm, bias=False)
        )

    def forward(self, input_ids, attention_mask, labels, pixel_values, **kwargs):
        B, L = input_ids.shape
        # Simulate merger output to trigger forward hook registered by DualObjectiveModel
        dummy_tokens = torch.randn(B, 6, self.d_lm)
        _ = self.visual.merger(dummy_tokens)  # fires the hook
        return _FakeOutput(B, L, self.V)


def _make_run_config(*, contrastive_enabled: bool = True):
    """Build a minimal RunConfig-like namespace without Pydantic validation."""
    model_cfg = types.SimpleNamespace(
        name="fake/model",
        revision=None,
        torch_dtype="bf16",
        attn_implementation="sdpa",
        freeze_vit=True,
        freeze_merger=True,
        unfreeze_vit_last_n=0,
    )
    lora_cfg = types.SimpleNamespace(
        enabled=False,
        r=4, alpha=8, dropout=0.0, target_modules=["q_proj"], bias="none"
    )
    aux_cfg = types.SimpleNamespace(
        proj_dim=32,
        pooler_num_heads=4,
        proj_hidden=64,
        chinese_clip_model="fake/clip",
    )
    contrastive_cfg = types.SimpleNamespace(
        enabled=contrastive_enabled,
        tau=0.07,
    )
    optim_cfg = types.SimpleNamespace(
        lr_lora=1e-4, weight_decay_lora=0.01,
        lr_aux=1e-3, weight_decay_aux=0.0, weight_decay_tag_proj=0.0,
    )
    return types.SimpleNamespace(
        model=model_cfg,
        lora=lora_cfg,
        aux=aux_cfg,
        contrastive=contrastive_cfg,
        optim=optim_cfg,
    )


def _make_vocab(tmp_path, n_tags=8):
    import json
    from formosa_dual.data.tag_vocab import TagVocabulary
    tags = [{"id": i, "tag": f"tag_{i}", "tier": 1, "freq": 5, "category": "test"} for i in range(n_tags)]
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps({"version": "v1", "size": n_tags, "tags": tags}), encoding="utf-8")
    return TagVocabulary(p)


# ---------------------------------------------------------------------------
# Stub DualObjectiveModel subclass that patches out heavy modules
# ---------------------------------------------------------------------------


def _build_stub_model(cfg, vocab, tmp_path):
    """Build a DualObjectiveModel with faked-out backbone and TagProjector."""
    import formosa_dual.models.dual_model as dm
    import formosa_dual.models.tag_projector as tp

    fake_backbone = _FakeBackbone(d_lm=64)
    fake_processor = MagicMock()

    # TagProjector mock: get_tag_embeddings returns zero tensor of correct shape
    class _FakeTagProjector(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
            self.projector = nn.Linear(32, cfg.aux.proj_dim)

        def __call__(self, tag_ids):
            B, P = tag_ids.shape
            return torch.zeros(B, P, cfg.aux.proj_dim)

    # Patch at the locations dual_model.py actually calls
    with (
        patch("formosa_dual.models.dual_model.load_backbone", return_value=(fake_backbone, fake_processor)),
        patch("formosa_dual.models.dual_model.apply_freeze_policy", return_value=None),
        patch("formosa_dual.models.tag_projector.TagProjector", _FakeTagProjector),
    ):
        model = dm.DualObjectiveModel(cfg, vocab, fake_processor)

    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def model_with_contrastive(tmp_path):
    cfg = _make_run_config(contrastive_enabled=True)
    vocab = _make_vocab(tmp_path)
    return _build_stub_model(cfg, vocab, tmp_path), cfg


@pytest.fixture
def model_no_contrastive(tmp_path):
    cfg = _make_run_config(contrastive_enabled=False)
    vocab = _make_vocab(tmp_path)
    return _build_stub_model(cfg, vocab, tmp_path), cfg


def _make_batch(B=2, L=8, V=1000, P=3, M=4, device="cpu"):
    return {
        "input_ids": torch.randint(0, V, (B, L)),
        "attention_mask": torch.ones(B, L, dtype=torch.long),
        "labels": torch.randint(-100, V, (B, L)),
        "pixel_values": torch.zeros(B, 3, 4, 4),
        "pos_tag_ids": torch.randint(0, 8, (B, P)),
        "pos_tag_mask": torch.ones(B, P, dtype=torch.bool),
        "neg_tag_ids": torch.randint(0, 8, (B, M)),
    }


def test_forward_keys_present(model_with_contrastive):
    model, _ = model_with_contrastive
    batch = _make_batch()
    out = model(batch)
    for key in ("lm_logits", "lm_loss", "visual_emb", "tag_pos_emb", "tag_neg_emb", "pos_tag_mask"):
        assert key in out, f"Missing key: {key}"


def test_forward_contrastive_none_when_disabled(model_no_contrastive):
    model, _ = model_no_contrastive
    batch = _make_batch()
    out = model(batch)
    assert out["visual_emb"] is None
    assert out["tag_pos_emb"] is None
    assert out["tag_neg_emb"] is None


def test_forward_visual_emb_shape(model_with_contrastive):
    model, cfg = model_with_contrastive
    batch = _make_batch(B=3)
    out = model(batch)
    assert out["visual_emb"] is not None
    assert out["visual_emb"].shape == (3, cfg.aux.proj_dim)


def test_get_trainable_param_groups_returns_named_groups(model_with_contrastive):
    model, _ = model_with_contrastive
    groups = model.get_trainable_param_groups()
    names = {g["name"] for g in groups}
    # At minimum aux and tag_proj should be present (no real LoRA params in stub)
    assert "aux" in names or "tag_proj" in names


def test_lm_loss_is_scalar(model_with_contrastive):
    model, _ = model_with_contrastive
    batch = _make_batch()
    out = model(batch)
    assert out["lm_loss"].shape == ()
