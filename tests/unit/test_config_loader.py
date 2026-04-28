"""Unit tests for formosa_dual.config.loader (§8.2)."""
import pytest

from formosa_dual.config import ConfigError, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_PATH = "configs"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_base_only():
    """Loading dev_smoke / v3_hero resolves a valid RunConfig."""
    cfg = load_config("dev_smoke", "v3_hero", base_path=BASE_PATH)
    assert cfg.experiment == "v3_hero"
    assert cfg.profile == "dev_smoke"
    assert cfg.model.name == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_load_with_profile_overrides_base():
    """Profile values override base values."""
    cfg = load_config("dev_mac", "v3_hero", base_path=BASE_PATH)
    # dev_mac.yaml sets per_device_batch_size=1
    assert cfg.training.per_device_batch_size == 1
    # dev_mac.yaml sets device.force=mps
    assert cfg.device.force == "mps"


def test_load_with_experiment_overrides_profile():
    """Experiment values override profile values."""
    # v1_caption_only disables contrastive
    cfg = load_config("dev_smoke", "v1_caption_only", base_path=BASE_PATH)
    assert cfg.contrastive.enabled is False
    assert cfg.caption.enabled is True


def test_cli_overrides_have_highest_priority():
    """CLI overrides win over YAML config chain."""
    cfg = load_config(
        "dev_smoke",
        "v3_hero",
        cli_overrides=["contrastive.tau=0.99"],
        base_path=BASE_PATH,
    )
    assert abs(cfg.contrastive.tau - 0.99) < 1e-9


def test_unknown_key_raises():
    """An unknown top-level key in CLI override raises ConfigError."""
    with pytest.raises(ConfigError):
        load_config(
            "dev_smoke",
            "v3_hero",
            cli_overrides=["nonexistent_key=123"],
            base_path=BASE_PATH,
        )


def test_smoke_flag_forces_smoke_enabled():
    """Passing smoke=True sets smoke.enabled=True regardless of YAML."""
    cfg = load_config("prod_gb10", "v3_hero", smoke=True, base_path=BASE_PATH)
    assert cfg.smoke.enabled is True


def test_at_least_one_loss_validator():
    """Disabling both losses raises ConfigError via the model validator."""
    with pytest.raises(ConfigError):
        load_config("dev_smoke", "v3_hero",
                    cli_overrides=["caption.enabled=false",
                                   "contrastive.enabled=false"],
                    base_path=BASE_PATH)
