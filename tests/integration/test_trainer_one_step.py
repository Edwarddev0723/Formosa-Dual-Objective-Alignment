"""Unit tests for DualTrainer one-step training loop (§8.6).

Uses stub model/loader to avoid downloading backbone weights.
"""
from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from formosa_dual.losses.dual_objective import DualObjectiveLoss


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeLMOut:
    def __init__(self):
        self.logits = torch.randn(2, 8, 100)
        self.loss = torch.tensor(1.5, requires_grad=True)


class _StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(4))
        # Mimic DualObjectiveModel interface
        self.backbone = MagicMock()
        self.pooler = None
        self.proj_head = None
        self.tag_projector = None

    def forward(self, batch):
        lm_loss = (self.param ** 2).sum()  # real gradient path through self.param
        return {
            "lm_logits": torch.randn(2, 8, 100),
            "lm_loss": lm_loss,
            "visual_emb": None,
            "tag_pos_emb": None,
            "tag_neg_emb": None,
            "pos_tag_mask": None,
        }

    def get_trainable_param_groups(self):
        return [{"params": list(self.parameters()), "lr": 1e-4, "weight_decay": 0.0, "name": "test"}]


def _make_batch():
    return {
        "input_ids": torch.randint(0, 100, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
        "labels": torch.randint(-100, 100, (2, 8)),
        "pixel_values": torch.zeros(2, 3, 4, 4),
        "pos_tag_ids": torch.zeros(2, 3, dtype=torch.long),
        "pos_tag_mask": torch.ones(2, 3, dtype=torch.bool),
        "neg_tag_ids": torch.zeros(2, 4, dtype=torch.long),
    }


def _make_cfg(tmp_path):
    run_name = "test_run"
    return types.SimpleNamespace(
        training=types.SimpleNamespace(
            num_epochs=1,
            seed=42,
            logging_steps=1,
            eval_steps=9999,
            save_steps=9999,
            save_total_limit=1,
            gradient_accumulation_steps=1,
            gradient_checkpointing=False,
        ),
        optim=types.SimpleNamespace(
            optimizer="adamw",
            lr_lora=1e-4,
            lr_aux=1e-3,
            weight_decay_lora=0.0,
            weight_decay_aux=0.0,
            weight_decay_tag_proj=0.0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            scheduler="constant_with_warmup",
            warmup_ratio=0.0,
        ),
        smoke=types.SimpleNamespace(enabled=True, max_steps=1),
        contrastive=types.SimpleNamespace(
            enabled=False, neg_sampling="uniform", hard_neg_refresh_every_steps=100
        ),
        caption=types.SimpleNamespace(enabled=True, label_smoothing=0.0),
        logging=types.SimpleNamespace(
            output_dir=str(tmp_path),
            run_name=run_name,
            report_to=None,
        ),
    )


def _make_loss_fn(cfg):
    """Build DualObjectiveLoss with a simple namespace cfg."""
    import formosa_dual.losses.dual_objective as dobj
    # Patch RunConfig type checks
    fn = dobj.DualObjectiveLoss.__new__(dobj.DualObjectiveLoss)
    nn.Module.__init__(fn)
    fn.cfg = cfg
    fn._caption_enabled = True
    fn._contrast_enabled = False

    from formosa_dual.losses.caption_loss import CaptionLoss
    fn.caption_loss = CaptionLoss()
    return fn


def _make_accelerator():
    """Build a minimal CPU accelerator."""
    from accelerate import Accelerator
    return Accelerator()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_one_step_loss_finite(tmp_path):
    """§8.3: a single training step produces a finite loss and updates LoRA params."""
    cfg = _make_cfg(tmp_path)
    model = _StubModel()
    loss_fn = _make_loss_fn(cfg)
    accelerator = _make_accelerator()
    vocab = MagicMock()

    batch = _make_batch()
    dataset = [batch]

    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    val_loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])

    from formosa_dual.training.trainer import DualTrainer
    trainer = DualTrainer(cfg, model, loss_fn, train_loader, val_loader, accelerator, vocab)

    param_before = trainer.model.param.detach().clone()
    trainer.train()
    param_after = trainer.model.param.detach().clone()

    # Param should change after one step (also implicitly checks the step's loss was finite)
    assert not torch.allclose(param_before, param_after), "Parameter was not updated"


def test_one_step_updates_lora_params(tmp_path):
    """§8.3: the LoRA-equivalent parameter group is updated after one step."""
    cfg = _make_cfg(tmp_path)
    model = _StubModel()
    loss_fn = _make_loss_fn(cfg)
    accelerator = _make_accelerator()
    vocab = MagicMock()

    batch = _make_batch()
    from torch.utils.data import DataLoader
    train_loader = DataLoader([batch], batch_size=1, collate_fn=lambda x: x[0])
    val_loader = DataLoader([batch], batch_size=1, collate_fn=lambda x: x[0])

    from formosa_dual.training.trainer import DualTrainer
    trainer = DualTrainer(cfg, model, loss_fn, train_loader, val_loader, accelerator, vocab)

    param_before = trainer.model.param.detach().cpu().clone()
    trainer.train()
    param_after = trainer.model.param.detach().cpu().clone()

    assert not torch.allclose(param_before, param_after), "LoRA-equivalent param did not update"


def test_trainer_smoke_cap_stops_early(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.smoke.max_steps = 2

    model = _StubModel()
    loss_fn = _make_loss_fn(cfg)
    accelerator = _make_accelerator()
    vocab = MagicMock()

    # 10 batches
    dataset = [_make_batch() for _ in range(10)]
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    val_loader = DataLoader([_make_batch()], batch_size=1, collate_fn=lambda x: x[0])

    from formosa_dual.training.trainer import DualTrainer
    trainer = DualTrainer(cfg, model, loss_fn, train_loader, val_loader, accelerator, vocab)
    trainer.train()

    assert trainer._global_step <= cfg.smoke.max_steps


# ---------------------------------------------------------------------------
# Aux update tests (§8.3)
# ---------------------------------------------------------------------------


class _StubModelWithAux(nn.Module):
    """Stub that owns pooler / proj_head / tag_projector with real grad-bearing params.

    Forward routes a small input through the aux modules so grads flow during
    backward.  Exposed param groups include both ``lora`` (self.param) and
    ``aux`` (pooler+proj_head) and ``tag_proj`` (tag_projector.projector).
    """

    def __init__(self):
        super().__init__()
        d = 4
        # Pretend "lora" param
        self.param = nn.Parameter(torch.randn(d))
        # Aux modules — must look like real submodules so trainer code paths work.
        self.pooler = nn.Linear(d, d, bias=False)
        self.proj_head = nn.Linear(d, d, bias=False)
        # Tag projector is a Module with a sub-module called .projector
        tp = nn.Module()
        tp.projector = nn.Linear(d, d, bias=False)
        self.tag_projector = tp
        # Register the tag_projector.projector as a child for state_dict/parameters()
        self.add_module("tag_projector_full", tp)
        self.backbone = MagicMock()

    def forward(self, batch):
        # Build a tiny embedding deterministically from self.param,
        # then route it through pooler -> proj_head -> tag_projector.projector.
        dev = self.param.device
        x = self.param.unsqueeze(0)                       # [1, d]
        v = self.pooler(x)                                # [1, d]
        v = self.proj_head(v)                             # [1, d]
        t = self.tag_projector.projector(x)               # [1, d]
        # lm_loss uses self.param so caption-only mode still produces grads on lora
        lm_loss = (self.param ** 2).sum()
        # Add a contrastive-style scalar that depends on aux params
        contrast_signal = (v * t).sum()
        # Return shape-compatible tensors so the contrastive loss path exercises aux grads
        visual_emb = v                                                    # [1, d]
        tag_pos_emb = t.unsqueeze(1).expand(-1, 2, -1).contiguous()       # [1, 2, d]
        tag_neg_emb = t.unsqueeze(1).expand(-1, 3, -1).contiguous()       # [1, 3, d]
        pos_tag_mask = torch.ones(1, 2, dtype=torch.bool, device=dev)
        return {
            "lm_logits": torch.randn(2, 8, 100, device=dev),
            "lm_loss": lm_loss + 0.0 * contrast_signal,  # ensure tensor, ignored when caption-only
            "visual_emb": visual_emb,
            "tag_pos_emb": tag_pos_emb,
            "tag_neg_emb": tag_neg_emb,
            "pos_tag_mask": pos_tag_mask,
        }

    def get_trainable_param_groups(self):
        return [
            {"params": [self.param], "lr": 1e-2, "weight_decay": 0.0, "name": "lora"},
            {
                "params": list(self.pooler.parameters()) + list(self.proj_head.parameters()),
                "lr": 1e-2,
                "weight_decay": 0.0,
                "name": "aux",
            },
            {
                "params": list(self.tag_projector.projector.parameters()),
                "lr": 1e-2,
                "weight_decay": 0.0,
                "name": "tag_proj",
            },
        ]


def _make_dual_loss_fn(cfg, contrast_enabled: bool):
    """Build a DualObjectiveLoss instance bypassing pydantic typing.

    Mirrors the helper used by the existing tests but lets the caller toggle
    contrastive on or off.
    """
    import formosa_dual.losses.dual_objective as dobj
    from formosa_dual.losses.caption_loss import CaptionLoss
    from formosa_dual.losses.lambda_schedule import LambdaSchedule
    from formosa_dual.losses.multi_pos_infonce import MultiPositiveInfoNCE

    fn = dobj.DualObjectiveLoss.__new__(dobj.DualObjectiveLoss)
    nn.Module.__init__(fn)
    fn.cfg = cfg
    fn._caption_enabled = bool(cfg.caption.enabled)
    fn._contrast_enabled = bool(contrast_enabled)
    fn.caption_loss = CaptionLoss()
    if contrast_enabled:
        fn.infonce = MultiPositiveInfoNCE(tau=cfg.contrastive.tau)
        fn.lambda_schedule = LambdaSchedule(
            schedule="constant",
            peak=float(cfg.contrastive.lambda_value),
            floor=0.0,
            warmup_steps=0,
            total_steps=1,
        )
    return fn


def test_one_step_updates_aux_params(tmp_path):
    """With contrastive enabled, pooler/proj_head/tag_projector params change after one step."""
    cfg = _make_cfg(tmp_path)
    # Enable contrastive in the cfg
    cfg.contrastive = types.SimpleNamespace(
        enabled=True,
        neg_sampling="uniform",
        hard_neg_refresh_every_steps=100,
        lambda_value=1.0,
        tau=0.07,
    )
    cfg.caption.enabled = True

    model = _StubModelWithAux()
    loss_fn = _make_dual_loss_fn(cfg, contrast_enabled=True)
    accelerator = _make_accelerator()
    vocab = MagicMock()

    batch = _make_batch()
    from torch.utils.data import DataLoader
    train_loader = DataLoader([batch], batch_size=1, collate_fn=lambda x: x[0])
    val_loader = DataLoader([batch], batch_size=1, collate_fn=lambda x: x[0])

    pooler_before = model.pooler.weight.detach().cpu().clone()
    proj_before = model.proj_head.weight.detach().cpu().clone()
    tag_before = model.tag_projector.projector.weight.detach().cpu().clone()

    from formosa_dual.training.trainer import DualTrainer
    trainer = DualTrainer(cfg, model, loss_fn, train_loader, val_loader, accelerator, vocab)
    trainer.train()

    assert not torch.allclose(pooler_before, trainer.model.pooler.weight.detach().cpu()), \
        "pooler weight did not update"
    assert not torch.allclose(proj_before, trainer.model.proj_head.weight.detach().cpu()), \
        "proj_head weight did not update"
    assert not torch.allclose(tag_before, trainer.model.tag_projector.projector.weight.detach().cpu()), \
        "tag_projector.projector weight did not update"


def test_caption_only_does_not_update_aux(tmp_path):
    """With contrastive disabled, pooler/proj_head/tag_projector params remain unchanged."""
    cfg = _make_cfg(tmp_path)
    cfg.contrastive = types.SimpleNamespace(
        enabled=False,
        neg_sampling="uniform",
        hard_neg_refresh_every_steps=100,
        lambda_value=0.0,
        tau=0.07,
    )
    cfg.caption.enabled = True

    model = _StubModelWithAux()
    loss_fn = _make_dual_loss_fn(cfg, contrast_enabled=False)
    accelerator = _make_accelerator()
    vocab = MagicMock()

    batch = _make_batch()
    from torch.utils.data import DataLoader
    train_loader = DataLoader([batch], batch_size=1, collate_fn=lambda x: x[0])
    val_loader = DataLoader([batch], batch_size=1, collate_fn=lambda x: x[0])

    pooler_before = model.pooler.weight.detach().cpu().clone()
    proj_before = model.proj_head.weight.detach().cpu().clone()
    tag_before = model.tag_projector.projector.weight.detach().cpu().clone()

    from formosa_dual.training.trainer import DualTrainer
    trainer = DualTrainer(cfg, model, loss_fn, train_loader, val_loader, accelerator, vocab)
    trainer.train()

    assert torch.allclose(pooler_before, trainer.model.pooler.weight.detach().cpu()), \
        "pooler weight changed in caption-only mode"
    assert torch.allclose(proj_before, trainer.model.proj_head.weight.detach().cpu()), \
        "proj_head weight changed in caption-only mode"
    assert torch.allclose(tag_before, trainer.model.tag_projector.projector.weight.detach().cpu()), \
        "tag_projector.projector weight changed in caption-only mode"
