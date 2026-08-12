"""Command-line interface for the OCR-RAG service."""

from __future__ import annotations

import argparse
from pathlib import Path

from .service import answer_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Question about the indexed documents")
    parser.add_argument("--index", type=Path, default=Path("outputs/rag_index.json"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--show-context", action="store_true", help="Print retrieved OCR chunks"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = answer_question(args.question, args.index, args.top_k, args.model)

    if args.show_context:
        for rank, context in enumerate(result["contexts"], start=1):
            print(
                f"\n--- Retrieved {rank} | similarity={context['similarity']:.4f} "
                f"{context['citation']} ---"
            )
            print(context["text"])

    print("\nAnswer")
    print(result["answer"])


if __name__ == "__main__":
    main()
