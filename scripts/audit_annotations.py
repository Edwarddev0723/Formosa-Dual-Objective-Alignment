#!/usr/bin/env python
"""scripts/audit_annotations.py — sample-based annotation audit (§6.4).

Usage:
    python scripts/audit_annotations.py \\
        --annotations PATH \\
        --sample-size 200 \\
        --output PATH  (must end in .html)

    # After manual review:
    python scripts/audit_annotations.py \\
        --annotations PATH \\
        --score-from PATH.html \\
        --output PATH.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit annotation quality via an HTML review page.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--output", required=True, help="Output HTML path.")
    parser.add_argument("--score-from", default=None, help="Reviewed HTML to score.")
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.suffix == ".html":
        logger.error("--output must end in .html")
        sys.exit(1)

    if args.score_from:
        _score_from_html(Path(args.score_from))
    else:
        _generate_audit_html(Path(args.annotations), args.sample_size, output_path)

    sys.exit(0)


def _generate_audit_html(annotations_path: Path, sample_size: int, output_path: Path) -> None:
    import json
    import random

    from formosa_dual.data.manifest import load_manifest

    records = load_manifest(annotations_path)
    sample = random.sample(records, min(sample_size, len(records)))
    logger.info("Generating audit HTML for %d records → %s", len(sample), output_path)

    rows = []
    for rec in sample:
        rid = rec.get("id", "")
        caption = rec.get("caption", "")
        tags = rec.get("culture_tags", [])
        image_path = rec.get("image_path", "")
        tag_checkboxes = " ".join(
            f'<label><input type="checkbox" name="{rid}_{t}" value="{t}" checked> {t}</label>'
            for t in tags
        )
        rows.append(
            f"<tr><td>{rid}</td>"
            f"<td><img src='{image_path}' style='max-width:200px'></td>"
            f"<td>{caption}</td>"
            f"<td>{tag_checkboxes}</td></tr>"
        )

    html = (
        "<!DOCTYPE html><html><body>"
        "<h1>Annotation Audit</h1>"
        "<form method='POST'>"
        "<table border='1'><tr><th>ID</th><th>Image</th><th>Caption</th><th>Tags</th></tr>"
        + "\n".join(rows)
        + "</table>"
        "<button type='submit'>Save</button></form></body></html>"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Audit HTML written to %s", output_path)


def _score_from_html(html_path: Path) -> None:
    """Parse reviewed HTML and report precision."""
    # ASSUMPTION: This is a simplified scorer; a full implementation would
    # parse checkbox states from the saved HTML form.
    logger.warning("Scoring from HTML not fully implemented; manual parsing required.")


if __name__ == "__main__":
    main()
