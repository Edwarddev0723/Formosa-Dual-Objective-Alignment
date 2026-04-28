"""Unit tests for LambdaSchedule (§8.5)."""
import math
import pytest
from formosa_dual.losses.lambda_schedule import LambdaSchedule


def test_constant_always_returns_peak():
    sched = LambdaSchedule("constant", peak=0.5)
    for step in [0, 1, 100, 9999]:
        assert sched(step) == pytest.approx(0.5)


def test_warmup_zero_at_start():
    sched = LambdaSchedule("warmup", peak=1.0, warmup_steps=100)
    assert sched(0) == pytest.approx(0.0)


def test_warmup_reaches_peak_at_warmup_steps():
    sched = LambdaSchedule("warmup", peak=1.0, warmup_steps=100)
    assert sched(100) == pytest.approx(1.0)


def test_warmup_stays_at_peak_after_warmup():
    sched = LambdaSchedule("warmup", peak=0.3, warmup_steps=50)
    for step in [50, 100, 200]:
        assert sched(step) == pytest.approx(0.3)


def test_warmup_anneal_reaches_floor_at_end():
    sched = LambdaSchedule(
        "warmup_anneal",
        peak=1.0,
        floor=0.0,
        warmup_steps=10,
        total_steps=100,
        anneal_ratio=0.5,
    )
    # At total_steps-1 (last step), cosine anneal should be near floor
    lam = sched(99)
    assert lam == pytest.approx(0.0, abs=0.05)


def test_warmup_anneal_plateau_between_warmup_and_anneal():
    sched = LambdaSchedule(
        "warmup_anneal",
        peak=0.8,
        floor=0.0,
        warmup_steps=10,
        total_steps=100,
        anneal_ratio=0.3,
    )
    # Step 50 is in the plateau (anneal_start = 70)
    assert sched(50) == pytest.approx(0.8)


def test_invalid_schedule_raises():
    with pytest.raises(ValueError):
        LambdaSchedule("invalid", peak=1.0)
