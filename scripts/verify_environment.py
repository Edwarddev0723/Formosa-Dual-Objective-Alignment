#!/usr/bin/env python
"""scripts/verify_environment.py — check that the runtime environment is complete.

Exit codes:
  0 — all critical checks pass
  1 — non-critical missing (no flash-attn, bitsandbytes, wandb — OK on Mac)
  2 — critical missing (no torch, no transformers)
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the formosa-dual runtime environment.")
    parser.add_argument("--verbose", action="store_true", help="Show extra detail.")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON to stdout.")
    args = parser.parse_args()

    from formosa_dual.training.device import device_capability_report
    from formosa_dual.utils.logging import get_logger

    logger = get_logger(__name__)
    report = device_capability_report()

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_table(report, verbose=args.verbose)

    # Critical checks
    critical_ok = all([
        report.get("torch_version") not in (None, ""),
        report.get("transformers_version") not in (None, ""),
    ])
    if not critical_ok:
        logger.error("Critical dependencies missing (torch or transformers).")
        sys.exit(2)

    # Non-critical
    non_critical_warn = any([
        not report.get("has_flash_attn", False),
        not report.get("has_bitsandbytes", False),
    ])
    if non_critical_warn:
        logger.warning("Some optional dependencies are missing (flash-attn, bitsandbytes). OK on Mac.")
        sys.exit(1)

    sys.exit(0)


def _print_table(report: dict, verbose: bool) -> None:
    print("\n=== Formosa-Dual Environment Report ===\n")
    for k, v in report.items():
        if not verbose and k.startswith("_"):
            continue
        print(f"  {k:<40} {v}")
    print()


if __name__ == "__main__":
    main()
