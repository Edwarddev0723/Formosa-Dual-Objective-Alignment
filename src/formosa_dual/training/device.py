"""formosa_dual.training.device — device selection, dtype mapping, and capability report.

Provides:
    - :func:`select_device` — pick the best available device.
    - :func:`get_supported_dtype` — map requested precision to torch dtype.
    - :func:`has_bitsandbytes` / :func:`has_flash_attn` — optional dep probes.
    - :func:`device_capability_report` — structured dict for diagnostics.

None of the optional imports (bitsandbytes, flash_attn) occur at module top level.
"""
from __future__ import annotations

import platform
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch

from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Optional dependency probes
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def has_bitsandbytes() -> bool:
    """Probe whether bitsandbytes is importable.  Result is cached."""
    try:
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def has_flash_attn() -> bool:
    """Probe whether flash_attn is importable.  Result is cached."""
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def select_device(cfg) -> torch.device:
    """Select the compute device based on ``cfg.force`` or auto-detection.

    Priority when auto-detecting: CUDA > MPS > CPU.

    Args:
        cfg: A :class:`~formosa_dual.config.schema.DeviceConfig` instance
            (or any object with ``force`` and ``auto_detect`` attributes).

    Returns:
        Selected :class:`torch.device`.
    """
    if cfg.force is not None:
        device = torch.device(cfg.force)
        logger.info("Device forced to %s via config", device)
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Auto-detected device: %s", device)
    return device


# ---------------------------------------------------------------------------
# dtype mapping
# ---------------------------------------------------------------------------


def get_supported_dtype(
    device: torch.device,
    requested: Literal["bf16", "fp16", "fp32", "no"],
) -> torch.dtype:
    """Map a requested precision string to a :class:`torch.dtype`.

    Device-aware fallbacks:
    - ``bf16`` on MPS: requires Apple Silicon + macOS 14+; falls back to fp32.
    - ``bf16`` on CUDA: always supported on modern hardware.
    - ``bf16`` on CPU: returns ``float32`` (not reliably supported).
    - ``"no"`` always returns ``float32``.

    Args:
        device: The selected compute device.
        requested: Precision string.

    Returns:
        Resolved :class:`torch.dtype`.
    """
    if requested in ("fp32", "no"):
        return torch.float32

    if requested == "fp16":
        if device.type == "cpu":
            logger.warning("fp16 on CPU unreliable; using float32")
            return torch.float32
        return torch.float16

    # requested == "bf16"
    if device.type == "cuda":
        return torch.bfloat16

    if device.type == "mps":
        # Check macOS version >= 14 and that MPS bf16 is available
        mac_ver = platform.mac_ver()[0]
        try:
            major = int(mac_ver.split(".")[0])
        except (ValueError, IndexError):
            major = 0
        if major >= 14 and torch.backends.mps.is_available():
            return torch.bfloat16
        logger.warning(
            "bf16 on MPS requires macOS 14+ (detected %s); falling back to float32",
            mac_ver,
        )
        return torch.float32

    # CPU
    logger.warning("bf16 not supported on CPU; using float32")
    return torch.float32


# ---------------------------------------------------------------------------
# Capability report
# ---------------------------------------------------------------------------


def _disk_free_gb(path: Path) -> float:
    """Return free disk space in GB at *path*, or -1.0 on error."""
    try:
        usage = shutil.disk_usage(path)
        return round(usage.free / (1024**3), 2)
    except OSError:
        return -1.0


def _hf_cache_size_gb(hf_home: Path) -> float:
    """Estimate size of HF cache directory in GB."""
    try:
        total = sum(f.stat().st_size for f in hf_home.rglob("*") if f.is_file())
        return round(total / (1024**3), 2)
    except (OSError, PermissionError):
        return -1.0


def device_capability_report() -> dict:
    """Return a structured capability report dict.

    Keys:
        python_version, torch_version, torch_cuda_available, torch_mps_available,
        device, compute_capability (if cuda), bf16_supported,
        bitsandbytes_available, flash_attn_available,
        transformers_version, peft_version, accelerate_version,
        free_disk_gb_at_data, free_disk_gb_at_outputs,
        hf_cache_path, hf_cache_size_gb.

    Returns:
        dict with the above keys.
    """
    import os

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    if cuda_available:
        device_str = "cuda"
    elif mps_available:
        device_str = "mps"
    else:
        device_str = "cpu"

    compute_cap: str | None = None
    bf16_supported = False

    if cuda_available:
        cc = torch.cuda.get_device_capability()
        compute_cap = f"{cc[0]}.{cc[1]}"
        bf16_supported = True
    elif mps_available:
        mac_ver = platform.mac_ver()[0]
        try:
            major = int(mac_ver.split(".")[0])
        except (ValueError, IndexError):
            major = 0
        bf16_supported = major >= 14

    def _pkg_version(pkg: str) -> str:
        try:
            import importlib.metadata

            return importlib.metadata.version(pkg)
        except Exception:
            return "unknown"

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))

    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_available": cuda_available,
        "torch_mps_available": mps_available,
        "device": device_str,
        "compute_capability": compute_cap,
        "bf16_supported": bf16_supported,
        "bitsandbytes_available": has_bitsandbytes(),
        "flash_attn_available": has_flash_attn(),
        "transformers_version": _pkg_version("transformers"),
        "peft_version": _pkg_version("peft"),
        "accelerate_version": _pkg_version("accelerate"),
        "free_disk_gb_at_data": _disk_free_gb(Path("data")),
        "free_disk_gb_at_outputs": _disk_free_gb(Path("outputs")),
        "hf_cache_path": str(hf_home),
        "hf_cache_size_gb": _hf_cache_size_gb(hf_home),
    }
