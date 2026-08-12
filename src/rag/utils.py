"""Utilities for turning OCR lines into searchable RAG chunks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable


def load_ocr_documents(report_path: Path, limit: int | None = None) -> list[dict]:
    """Load ordered OCR prediction lines from an evaluation report."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    documents = report.get("documents", [])
    if limit is not None:
        documents = documents[:limit]

    loaded: list[dict] = []
    for document in documents:
        assignments = sorted(
            document.get("line_assignments", []),
            key=lambda item: item["prediction_index"],
        )
        lines = [
            item["prediction_text"].strip()
            for item in assignments
            if item.get("prediction_text", "").strip()
        ]
        if lines:
            loaded.append({"source": document["image"], "lines": lines})
    return loaded


def chunk_document(
    source: str,
    lines: list[str],
    target_characters: int = 700,
    overlap_lines: int = 2,
) -> list[dict]:
    """Group complete OCR lines into overlapping, source-aware chunks."""
    if target_characters <= 0:
        raise ValueError("target_characters must be positive")
    if overlap_lines < 0:
        raise ValueError("overlap_lines cannot be negative")

    chunks: list[dict] = []
    start = 0
    while start < len(lines):
        end = start
        length = 0
        while end < len(lines):
            added = len(lines[end]) + (1 if end > start else 0)
            if end > start and length + added > target_characters:
                break
            length += added
            end += 1

        chunks.append(
            {
                "id": f"{Path(source).stem}-chunk-{len(chunks):03d}",
                "source": source,
                "line_start": start,
                "line_end": end - 1,
                "text": "\n".join(lines[start:end]),
            }
        )
        if end >= len(lines):
            break
        start = max(start + 1, end - overlap_lines)
    return chunks


def build_chunks(
    documents: Iterable[dict],
    target_characters: int = 700,
    overlap_lines: int = 2,
) -> list[dict]:
    chunks: list[dict] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document["source"],
                document["lines"],
                target_characters,
                overlap_lines,
            )
        )
    return chunks


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)

