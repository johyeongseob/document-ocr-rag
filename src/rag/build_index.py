"""Build an embedding index from OCR predictions in an evaluation report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from .utils import build_chunks, load_ocr_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", type=Path, default=Path("outputs/evaluation_all.json")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/rag_index.json"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--overlap-lines", type=int, default=2)
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set the OPENAI_API_KEY environment variable first.")
    if not args.report.is_file():
        raise FileNotFoundError(f"Evaluation report not found: {args.report}")

    documents = load_ocr_documents(args.report, args.limit)
    chunks = build_chunks(documents, args.chunk_size, args.overlap_lines)
    if not chunks:
        raise ValueError("No OCR text was found in the evaluation report.")

    client = OpenAI()
    response = client.embeddings.create(
        model=args.embedding_model,
        input=[chunk["text"] for chunk in chunks],
    )
    for chunk, embedding in zip(chunks, response.data):
        chunk["embedding"] = embedding.embedding

    index = {
        "embedding_model": args.embedding_model,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Index: {args.output.resolve()}")


if __name__ == "__main__":
    main()
