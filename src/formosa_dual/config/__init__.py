"""formosa_dual.config — configuration loading and validation."""
from formosa_dual.config.loader import load_config
from formosa_dual.config.schema import RunConfig


class ConfigError(Exception):
    """Raised on configuration validation or loading failures."""


__all__ = ["load_config", "RunConfig", "ConfigError"]
