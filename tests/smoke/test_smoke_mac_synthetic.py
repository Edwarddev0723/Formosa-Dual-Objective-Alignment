"""tests/smoke/test_smoke_mac_synthetic.py — full DualTrainer pipeline on synthetic stubs.

Runs in <60 seconds on Mac (CPU/MPS). Uses a small stub backbone to skip
HuggingFace downloads but exercises the real DualTrainer + Accelerator +
DualObjectiveLoss + LoggingCallback code paths.
"""
from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


@pytest.mark.smoke
def test_synthetic_smoke_runs_3_steps(tmp_path):
    """Full pipeline runs end-to-end on synthetic data for 3 steps."""
    from formosa_dual.utils.synthetic import make_synthetic_record
    from formosa_dual.data.tag_vocab import TagVocabulary
    from formosa_dual.utils.seeding import set_seed

    set_seed(42)

    # ------------------------------------------------------------------
    # Synthetic vocab + records (real primitives)
    # ------------------------------------------------------------------
    data_dir = tmp_path / "data"
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True)

    n_tags = 16
    tag_entries = [
        {"id": i, "tag": f"tag_{i}", "tier": 1, "freq": 5, "category": "test"}
        for i in range(n_tags)
    ]
    vocab_path = data_dir / "vocab.json"
    vocab_path.write_text(
        json.dumps({"version": "v1", "size": n_tags, "tags": tag_entries}),
        encoding="utf-8",
    )
    vocab = TagVocabulary(vocab_path)
    tag_strs = [t["tag"] for t in tag_entries]
    records = [make_synthetic_record(i, tag_strs, image_dir=img_dir) for i in range(8)]

    # ------------------------------------------------------------------
    # Minimal cfg (SimpleNamespace mirroring RunConfig surface)
    # ------------------------------------------------------------------
    cfg = types.SimpleNamespace(
        training=types.SimpleNamespace(
            num_epochs=1, seed=42, logging_steps=1,
            eval_steps=9999, save_steps=9999, save_total_limit=1,
            gradient_accumulation_steps=1, gradient_checkpointing=False,
        ),
        optim=types.SimpleNamespace(
            optimizer="adamw",
            lr_lora=1e-4, lr_aux=1e-3,
            weight_decay_lora=0.0, weight_decay_aux=0.0, weight_decay_tag_proj=0.0,
            adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8,
            max_grad_norm=1.0, scheduler="constant_with_warmup", warmup_ratio=0.0,
        ),
        smoke=types.SimpleNamespace(enabled=True, max_steps=3),
        contrastive=types.SimpleNamespace(
            enabled=False, neg_sampling="uniform", hard_neg_refresh_every_steps=100,
        ),
        caption=types.SimpleNamespace(enabled=True, label_smoothing=0.0),
        logging=types.SimpleNamespace(
            output_dir=str(tmp_path / "output"),
            run_name="smoke_test",
            report_to=None,
        ),
    )

    # ------------------------------------------------------------------
    # Stub model
    # ------------------------------------------------------------------
    class _StubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.randn(8))
            self.backbone = MagicMock()
            self.pooler = None
            self.proj_head = None
            self.tag_projector = None

        def forward(self, batch):
            return {
                "lm_logits": torch.randn(2, 4, 100),
                "lm_loss": (self.param ** 2).sum(),
                "visual_emb": None,
                "tag_pos_emb": None,
                "tag_neg_emb": None,
                "pos_tag_mask": None,
            }

        def get_trainable_param_groups(self):
            return [{"params": list(self.parameters()), "lr": 1e-4,
                     "weight_decay": 0.0, "name": "test"}]

    # ------------------------------------------------------------------
    # Loss fn (skip pydantic-typed __init__)
    # ------------------------------------------------------------------
    import formosa_dual.losses.dual_objective as dobj
    from formosa_dual.losses.caption_loss import CaptionLoss

    loss_fn = dobj.DualObjectiveLoss.__new__(dobj.DualObjectiveLoss)
    nn.Module.__init__(loss_fn)
    loss_fn.cfg = cfg
    loss_fn._caption_enabled = True
    loss_fn._contrast_enabled = False
    loss_fn.caption_loss = CaptionLoss()

    # ------------------------------------------------------------------
    # Stub batches (avoid loading real Qwen processor)
    # ------------------------------------------------------------------
    def _stub_collate(_):
        return {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "labels": torch.zeros(2, 4, dtype=torch.long) - 100,
            "pixel_values": torch.zeros(2, 3, 4, 4),
            "pos_tag_ids": torch.zeros(2, 3, dtype=torch.long),
            "pos_tag_mask": torch.ones(2, 3, dtype=torch.bool),
            "neg_tag_ids": torch.zeros(2, 4, dtype=torch.long),
        }

    from torch.utils.data import DataLoader
    train_loader = DataLoader(records, batch_size=2, collate_fn=_stub_collate)
    val_loader = DataLoader(records[:2], batch_size=2, collate_fn=_stub_collate)

    # ------------------------------------------------------------------
    # Run trainer
    # ------------------------------------------------------------------
    from accelerate import Accelerator
    from formosa_dual.training.trainer import DualTrainer

    accelerator = Accelerator()
    model = _StubModel()
    trainer = DualTrainer(cfg, model, loss_fn, train_loader, val_loader, accelerator, vocab)

    param_before = trainer.model.param.detach().clone()
    trainer.train()
    param_after = trainer.model.param.detach().clone()

    assert not torch.allclose(param_before, param_after), "Parameter must update"
    assert trainer._global_step <= cfg.smoke.max_steps
