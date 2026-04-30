"""Tests for scripts/annotate_tags.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def test_annotate_tags_script_writes_tag_strings(tmp_path, monkeypatch):
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(
        json.dumps(
            {
                "version": "v1",
                "size": 2,
                "tags": [
                    {"id": 0, "tag": "台灣", "tier": 1, "freq": 1, "category": "地理"},
                    {"id": 1, "tag": "廟宇", "tier": 1, "freq": 1, "category": "建築"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    input_path = tmp_path / "manifest.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "sample_1",
                "image_path": "unused.jpg",
                "caption": "台灣廟宇",
                "source": "test",
                "image_hash": "sha256:test",
                "difficulty": 3,
                "culture_tags": [],
                "metadata": {},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "annotated.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "annotate_tags.py",
            "--input",
            str(input_path),
            "--vocab",
            str(vocab_path),
            "--output",
            str(output_path),
            "--max-tags",
            "10",
        ],
    )

    from scripts.annotate_tags import main

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0

    record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["culture_tags"] == ["台灣", "廟宇"]
