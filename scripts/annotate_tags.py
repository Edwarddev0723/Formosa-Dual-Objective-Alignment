#!/usr/bin/env python
"""scripts/annotate_tags.py — annotate a manifest with cultural tags (§6.3).

Usage:
    python scripts/annotate_tags.py \\
        --input PATH \\
        --vocab PATH \\
        --use-aho-corasick \\
        --use-metadata \\
        [--use-llm MODEL_NAME] \\
        --max-tags 10 \\
        --output PATH \\
        --num-workers 4 \\
        [--resume]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate a JSONL manifest with cultural tags.")
    parser.add_argument("--input", required=True, help="Input JSONL manifest path.")
    parser.add_argument("--vocab", required=True, help="Tag vocabulary JSON path.")
    parser.add_argument("--use-aho-corasick", action="store_true", default=True)
    parser.add_argument("--use-metadata", action="store_true", default=True)
    parser.add_argument("--use-llm", default=None, help="HuggingFace model ID for LLM annotation.")
    parser.add_argument("--max-tags", type=int, default=10)
    parser.add_argument("--output", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Skip records already in output (matched by id).")
    args = parser.parse_args()

    from formosa_dual.data.manifest import load_manifest, write_manifest
    from formosa_dual.data.tag_annotator import TagAnnotator
    from formosa_dual.data.tag_vocab import TagVocabulary

    input_path = Path(args.input)
    output_path = Path(args.output)
    vocab_path = Path(args.vocab)

    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        sys.exit(1)
    if not vocab_path.exists():
        logger.error("Vocab not found: %s", vocab_path)
        sys.exit(1)

    vocab = TagVocabulary(vocab_path)
    annotator = TagAnnotator(
        vocab=vocab,
        use_aho_corasick=args.use_aho_corasick,
        use_metadata=args.use_metadata,
        max_tags=args.max_tags,
    )

    records = load_manifest(input_path)

    # Resume: skip already-annotated records
    done_ids: set[str] = set()
    existing: list[dict] = []
    if args.resume and output_path.exists():
        existing = load_manifest(output_path)
        done_ids = {r["id"] for r in existing}
        logger.info("Resuming: %d records already annotated", len(done_ids))

    pending = [r for r in records if r["id"] not in done_ids]
    logger.info("Annotating %d records (skipping %d)", len(pending), len(done_ids))

    annotated = []
    for i, record in enumerate(pending):
        tag_ids = annotator.annotate(record)
        tag_strs = [vocab.decode(tid) for tid in tag_ids if vocab.decode(tid)]
        record = dict(record)
        record["culture_tags"] = tag_strs
        annotated.append(record)
        if (i + 1) % 100 == 0:
            logger.info("Annotated %d / %d records", i + 1, len(pending))

    all_records = existing + annotated
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(all_records, output_path)
    logger.info("Saved %d annotated records to %s", len(all_records), output_path)
    sys.exit(0)


if __name__ == "__main__":
    main()
