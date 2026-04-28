"""tests/unit/test_dual_objective_loss_construction.py

Regression tests for `DualObjectiveLoss` construction against the real
`RunConfig` schema.

These tests catch the class of bug seen during the DoD §0.6 step-5 audit
where `DualObjectiveLoss.__init__` referenced `cfg.contrastive.lambda_peak`
and `cfg.contrastive.lambda_warmup_steps` while the schema (spec §4.1) only
defines `lambda_value` and `lambda_warmup_ratio`. The existing trainer
integration tests bypass this code path by constructing the loss via
``__new__`` and assigning ``infonce`` directly, so a field rename in either
the schema or the loss went undetected until full-model run time.

Every test below MUST instantiate ``DualObjectiveLoss(cfg=..., total_steps=...)``
through the normal constructor.
"""
from __future__ import annotations

import pytest

from formosa_dual.config.schema import (
    AuxModulesConfig,
    CaptionConfig,
    ContrastiveConfig,
    DataConfig,
    DeviceConfig,
    LoggingConfig,
    LoRAConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    SmokeConfig,
    TrainingConfig,
)
from formosa_dual.losses.dual_objective import DualObjectiveLoss


def _make_cfg(
    *,
    caption_enabled: bool = True,
    contrast_enabled: bool = True,
    lambda_schedule: str = "warmup",
    lambda_value: float = 0.2,
    lambda_warmup_ratio: float = 0.1,
    lambda_anneal_ratio: float = 0.0,
) -> RunConfig:
    return RunConfig(
        model=ModelConfig(name="Qwen/Qwen2.5-VL-3B-Instruct"),
        lora=LoRAConfig(),
        aux=AuxModulesConfig(),
        contrastive=ContrastiveConfig(
            enabled=contrast_enabled,
            lambda_schedule=lambda_schedule,
            lambda_value=lambda_value,
            lambda_warmup_ratio=lambda_warmup_ratio,
            lambda_anneal_ratio=lambda_anneal_ratio,
            tau=0.07,
        ),
        caption=CaptionConfig(enabled=caption_enabled, max_caption_tokens=64),
        data=DataConfig(
            train_manifest="data/synthetic/train_synth.jsonl",
            val_manifest="data/synthetic/val_synth.jsonl",
            vocab_path="data/synthetic/vocab_synth.json",
            image_root="data/synthetic/images",
        ),
        optim=OptimConfig(),
        training=TrainingConfig(per_device_batch_size=1, gradient_accumulation_steps=1),
        device=DeviceConfig(force="cpu", mixed_precision="no"),
        logging=LoggingConfig(),
        smoke=SmokeConfig(),
    )


def test_dual_loss_constructs_with_real_schema_fields():
    """DualObjectiveLoss must accept the canonical schema field names.

    Regression: `lambda_peak` / `lambda_warmup_steps` were used in the
    constructor; spec §4.1 names them `lambda_value` / `lambda_warmup_ratio`.
    """
    cfg = _make_cfg()
    loss = DualObjectiveLoss(cfg=cfg, total_steps=100)
    # Both branches must be wired
    assert hasattr(loss, "caption_loss")
    assert hasattr(loss, "infonce")
    assert hasattr(loss, "lambda_schedule")
    # Warmup ratio 0.1 over 100 steps → 10 warmup steps
    assert loss.lambda_schedule.warmup_steps == 10
    assert loss.lambda_schedule.peak == pytest.approx(0.2)


def test_dual_loss_caption_only_skips_infonce():
    cfg = _make_cfg(caption_enabled=True, contrast_enabled=False)
    loss = DualObjectiveLoss(cfg=cfg, total_steps=10)
    assert hasattr(loss, "caption_loss")
    assert not hasattr(loss, "infonce")
    assert not hasattr(loss, "lambda_schedule")


def test_dual_loss_contrast_only_skips_caption():
    cfg = _make_cfg(caption_enabled=False, contrast_enabled=True)
    loss = DualObjectiveLoss(cfg=cfg, total_steps=10)
    assert not hasattr(loss, "caption_loss")
    assert hasattr(loss, "infonce")
    assert hasattr(loss, "lambda_schedule")


def test_dual_loss_constant_schedule():
    cfg = _make_cfg(lambda_schedule="constant", lambda_value=0.5, lambda_warmup_ratio=0.0)
    loss = DualObjectiveLoss(cfg=cfg, total_steps=50)
    assert loss.lambda_schedule.peak == pytest.approx(0.5)
    # Constant schedule returns peak at every step
    assert loss.lambda_schedule(0) == pytest.approx(0.5)
    assert loss.lambda_schedule(25) == pytest.approx(0.5)


def test_dual_loss_warmup_anneal_schedule():
    cfg = _make_cfg(
        lambda_schedule="warmup_anneal",
        lambda_value=1.0,
        lambda_warmup_ratio=0.1,
        lambda_anneal_ratio=0.5,
    )
    loss = DualObjectiveLoss(cfg=cfg, total_steps=100)
    # Step 0 (warmup start) → small λ
    assert loss.lambda_schedule(0) < 1.0
    # Mid-plateau → at peak
    assert loss.lambda_schedule(40) == pytest.approx(1.0, abs=1e-3)
