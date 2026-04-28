#!/usr/bin/env python
"""scripts/build_tag_vocab.py — build the cultural tag vocabulary (§6.2).

Usage:
    python scripts/build_tag_vocab.py \\
        --tier1 PATH \\
        --tier2 PATH [PATH ...] \\
        --tier3-from-captions PATH \\
        --tier3-llm-model NAME \\
        --target-size 800 \\
        --min-freq 5 \\
        --output PATH \\
        [--smoke]
"""
from __future__ import annotations

import argparse
import sys

from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cultural tag vocabulary.")
    parser.add_argument("--tier1", required=True, help="Path to tier-1 tag list (text, one tag per line).")
    parser.add_argument("--tier2", nargs="+", default=[], help="Paths to tier-2 tag lists.")
    parser.add_argument("--tier3-from-captions", default=None, help="JSONL manifest for LLM-extracted tier-3 tags.")
    parser.add_argument("--tier3-llm-model", default=None, help="HuggingFace model ID for LLM tier-3 extraction.")
    parser.add_argument("--target-size", type=int, default=800, help="Target vocabulary size.")
    parser.add_argument("--min-freq", type=int, default=5, help="Minimum tag frequency in corpus.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--smoke", action="store_true", help="Smoke mode: tier-1 only, 50-tag output.")
    args = parser.parse_args()

    from pathlib import Path
    from formosa_dual.data.tag_vocab import TagVocabulary

    def _load_tag_list(path: str) -> list[str]:
        p = Path(path)
        if not p.exists():
            logger.error("Tag list not found: %s", p)
            sys.exit(1)
        return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    tier1_tags = _load_tag_list(args.tier1)
    logger.info("Loaded %d tier-1 tags", len(tier1_tags))

    if args.smoke:
        # Smoke: use only tier1, cap at 50
        tier1_tags = tier1_tags[:50]
        tier2_tags: list[str] = []
        tier3_tags: list[str] = []
        target_size = min(50, args.target_size)
        logger.info("Smoke mode: using only %d tier-1 tags.", len(tier1_tags))
    else:
        tier2_tags = []
        for p in args.tier2:
            tier2_tags.extend(_load_tag_list(p))
        logger.info("Loaded %d tier-2 tags", len(tier2_tags))

        tier3_tags = []
        if args.tier3_from_captions and Path(args.tier3_from_captions).exists():
            tier3_tags = _extract_tier3(args.tier3_from_captions, args.tier3_llm_model)
            logger.info("Extracted %d tier-3 tags", len(tier3_tags))

        target_size = args.target_size

    vocab = TagVocabulary.build(
        tier1=tier1_tags,
        tier2=tier2_tags,
        tier3=tier3_tags,
        freqs={},
        categories={},
        target_size=target_size,
        min_freq=0,  # freq filtering not done here (done by build with corpus)
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(output_path)
    logger.info("Vocabulary saved to %s (%d tags)", output_path, len(vocab))
    sys.exit(0)


def _extract_tier3(captions_path: str, llm_model: str | None) -> list[str]:
    """Extract tier-3 tags from a caption manifest using LLM."""
    if llm_model is None:
        logger.warning("--tier3-llm-model not specified; skipping tier-3 LLM extraction.")
        return []
    # ASSUMPTION: LLM extraction not implemented; requires a custom LLMClient subclass.
    logger.warning("LLM tier-3 extraction not implemented; returning empty list.")
    return []


if __name__ == "__main__":
    main()
