"""formosa_dual.config.loader — YAML config loading and merging.

Priority order (right wins, deep-merged):
    base.yaml < profiles/{profile}.yaml < experiments/{experiment}.yaml < cli_overrides

Lists are **replaced** entirely by the right-side value (not concatenated).
Dicts are recursively merged.

CLI overrides format: ``["key.subkey=value", ...]`` where values are parsed
as YAML scalars/structures so ``"0.5"`` → float, ``"true"`` → bool,
``"[a,b]"`` → list.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from formosa_dual.config.schema import RunConfig
from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*.

    Rules:
    - If both values are dicts: recurse.
    - Otherwise: override value replaces base value (lists included).
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _load_yaml(path: Path) -> dict:
    """Load a YAML file safely, raising :class:`ConfigError` on failure."""
    from formosa_dual.config import ConfigError

    if not path.exists():
        raise ConfigError(f"Config file not found: {path.resolve()}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error in {path}: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _apply_cli_overrides(data: dict, overrides: list[str]) -> dict:
    """Apply dotted ``key.subkey=value`` overrides onto *data* (in-place copy).

    Values are parsed with ``yaml.safe_load`` so ``"0.5"`` → float,
    ``"true"`` → bool, ``"[a,b]"`` → list, etc.

    Raises :class:`ConfigError` if an override string is malformed.
    """
    from formosa_dual.config import ConfigError

    result = copy.deepcopy(data)
    for override in overrides:
        if "=" not in override:
            raise ConfigError(
                f"Malformed CLI override (expected 'key=value'): {override!r}"
            )
        key_path, _, raw_value = override.partition("=")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Cannot parse value in override {override!r}: {exc}"
            ) from exc

        keys = key_path.strip().split(".")
        node = result
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    profile: str,
    experiment: str,
    cli_overrides: list[str] | None = None,
    smoke: bool = False,
    base_path: Path = Path("configs"),
    allow_no_loss: bool = False,
) -> RunConfig:
    """Load and merge configs in priority order (right wins, deep-merged).

    Priority: base.yaml < profiles/{profile}.yaml < experiments/{experiment}.yaml
              < cli_overrides

    Args:
        profile: Profile name (``dev_mac``, ``dev_smoke``, or ``prod_gb10``).
        experiment: Experiment name (e.g. ``v3_hero``) or sub-path for ablations
            (e.g. ``ablations/lambda_0.5``).
        cli_overrides: Optional list of ``"key.subkey=value"`` strings.
        smoke: If ``True``, force ``smoke.enabled = True`` regardless of config.
        base_path: Root directory for config files (default ``configs/``).
        allow_no_loss: If ``True``, bypass the ``at_least_one_loss_enabled``
            validator with a warning. **Only use for ``--dry-run``** — actual
            training with no loss is forbidden by the trainer entrypoint.

    Returns:
        Validated :class:`~formosa_dual.config.schema.RunConfig`.

    Raises:
        :class:`~formosa_dual.config.ConfigError`: On missing files, unknown keys,
            or schema violations.
    """
    from formosa_dual.config import ConfigError

    base_path = Path(base_path)

    # 1. Load base
    merged = _load_yaml(base_path / "base.yaml")
    logger.debug("Loaded base.yaml")

    # 2. Overlay profile
    profile_path = base_path / "profiles" / f"{profile}.yaml"
    profile_data = _load_yaml(profile_path)
    merged = _deep_merge(merged, profile_data)
    logger.debug("Merged profile %s", profile)

    # 3. Overlay experiment
    # Support sub-paths like "ablations/lambda_0.5"
    exp_path = base_path / "experiments" / f"{experiment}.yaml"
    exp_data = _load_yaml(exp_path)
    merged = _deep_merge(merged, exp_data)
    logger.debug("Merged experiment %s", experiment)

    # 4. Apply CLI overrides
    if cli_overrides:
        merged = _apply_cli_overrides(merged, cli_overrides)
        logger.debug("Applied %d CLI overrides", len(cli_overrides))

    # 5. Force smoke flag
    if smoke:
        smoke_section = merged.setdefault("smoke", {})
        smoke_section["enabled"] = True
        logger.debug("smoke.enabled forced True by --smoke flag")

    # 6. Stamp profile + experiment into the merged dict so RunConfig sees them
    merged.setdefault("profile", profile)
    merged.setdefault("experiment", experiment)
    merged["profile"] = profile
    merged["experiment"] = experiment

    # 7. Validate via pydantic (extra="forbid" to catch unknown keys)
    try:
        cfg = RunConfig.model_validate(merged)
    except Exception as exc:
        # Special-case: bypass the at_least_one_loss validator under allow_no_loss
        # (used only by --dry-run for v0_zero_shot).
        if allow_no_loss and "At least one of caption/contrastive" in str(exc):
            logger.warning(
                "Bypassing at_least_one_loss_enabled validator (allow_no_loss=True). "
                "This config is for --dry-run only and cannot start training."
            )
            # Build sub-configs strictly, then assemble the parent without
            # re-running the model_validator.
            from formosa_dual.config.schema import (
                AuxModulesConfig, CaptionConfig, ContrastiveConfig,
                DataConfig, DeviceConfig, LoggingConfig, LoRAConfig,
                ModelConfig, OptimConfig, SmokeConfig, TrainingConfig,
            )
            cfg = RunConfig.model_construct(
                profile=merged.get("profile", profile),
                experiment=merged.get("experiment", experiment),
                model=ModelConfig(**merged.get("model", {})),
                lora=LoRAConfig(**merged.get("lora", {})),
                aux=AuxModulesConfig(**merged.get("aux", {})),
                contrastive=ContrastiveConfig(**merged.get("contrastive", {})),
                caption=CaptionConfig(**merged.get("caption", {})),
                data=DataConfig(**merged.get("data", {})),
                optim=OptimConfig(**merged.get("optim", {})),
                training=TrainingConfig(**merged.get("training", {})),
                device=DeviceConfig(**merged.get("device", {})),
                logging=LoggingConfig(**merged.get("logging", {})),
                smoke=SmokeConfig(**merged.get("smoke", {})),
            )
        else:
            raise ConfigError(f"Config schema validation failed: {exc}") from exc

    logger.info("Config loaded: profile=%s experiment=%s", profile, experiment)
    return cfg
