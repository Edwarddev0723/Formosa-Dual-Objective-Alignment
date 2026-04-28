"""formosa_dual.eval.reporter — collect metric dicts and write reports (§5.22).

Writes ``report.json`` (machine-readable) and ``report.md`` (markdown table).
"""
from __future__ import annotations

import json
from pathlib import Path

from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


class Reporter:
    """Collect metric dicts and produce structured reports.

    Args:
        output_dir: Directory to write ``report.json`` and ``report.md``.
        run_name: Optional display name for the run.
    """

    def __init__(self, output_dir: Path, run_name: str = "run") -> None:
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self._sections: dict[str, dict] = {}

    def add_section(self, name: str, metrics: dict) -> None:
        """Add a named section of metrics.

        Args:
            name: Section name (e.g. ``"caption"``, ``"retrieval"``).
            metrics: Flat dict of metric name → scalar value.
        """
        self._sections[name] = metrics

    def write(self) -> tuple[Path, Path]:
        """Write ``report.json`` and ``report.md`` to *output_dir*.

        Returns:
            Tuple ``(json_path, md_path)``.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        payload = {"run": self.run_name, "sections": self._sections}
        json_path = self.output_dir / "report.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        # Markdown
        md_lines = [f"# Report: {self.run_name}", ""]
        for section, metrics in self._sections.items():
            md_lines.append(f"## {section}")
            md_lines.append("")
            md_lines.append("| Metric | Value |")
            md_lines.append("|---|---|")
            for k, v in metrics.items():
                if isinstance(v, float):
                    md_lines.append(f"| {k} | {v:.4f} |")
                else:
                    md_lines.append(f"| {k} | {v} |")
            md_lines.append("")

        md_path = self.output_dir / "report.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

        logger.info("Report written: %s, %s", json_path, md_path)
        return json_path, md_path
