#!/usr/bin/env python
"""scripts/download_models.py — pre-fetch HuggingFace models to local cache (§6.7).

Usage:
    python scripts/download_models.py \\
        --models MODEL_ID [MODEL_ID ...] \\
        [--cache-dir PATH]
"""
from __future__ import annotations

import argparse
import sys

from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-fetch HuggingFace models to local cache.")
    parser.add_argument("--models", nargs="+", required=True, help="Model IDs to download.")
    parser.add_argument("--cache-dir", default=None, help="Override HuggingFace cache directory.")
    args = parser.parse_args()

    errors = []
    for model_id in args.models:
        logger.info("Downloading: %s", model_id)
        try:
            from transformers import AutoConfig
            kwargs = {}
            if args.cache_dir:
                kwargs["cache_dir"] = args.cache_dir
            AutoConfig.from_pretrained(model_id, trust_remote_code=True, **kwargs)
            logger.info("OK: %s", model_id)
        except Exception as exc:
            logger.error("Failed to download %s: %s", model_id, exc)
            errors.append(model_id)

    if errors:
        logger.error("Failed to download %d model(s): %s", len(errors), errors)
        sys.exit(2)

    logger.info("All %d model(s) downloaded successfully.", len(args.models))
    sys.exit(0)


if __name__ == "__main__":
    main()
