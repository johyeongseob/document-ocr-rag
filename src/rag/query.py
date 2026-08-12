"""Retrieve OCR chunks and answer a question with cited evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from .utils import cosine_similarity


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
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set the OPENAI_API_KEY environment variable first.")
    if not args.index.is_file():
        raise FileNotFoundError(f"RAG index not found: {args.index}")

    index = json.loads(args.index.read_text(encoding="utf-8"))
    client = OpenAI()
    query_embedding = client.embeddings.create(
        model=index["embedding_model"], input=args.question
    ).data[0].embedding

    ranked = sorted(
        (
            (cosine_similarity(query_embedding, chunk["embedding"]), chunk)
            for chunk in index["chunks"]
        ),
        key=lambda item: item[0],
        reverse=True,
    )[: args.top_k]

    evidence_parts: list[str] = []
    for rank, (score, chunk) in enumerate(ranked, start=1):
        citation = f"[{chunk['source']}#{chunk['id']}]"
        evidence_parts.append(f"{citation}\n{chunk['text']}")
        if args.show_context:
            print(f"\n--- Retrieved {rank} | similarity={score:.4f} {citation} ---")
            print(chunk["text"])

    evidence = "\n\n".join(evidence_parts)
    prompt = f"""질문:
{args.question}

검색된 OCR 근거:
{evidence}

검색된 근거만 사용해 한국어로 간결하게 답하세요.
근거가 부족하면 추측하지 말고 '제공된 문서에서 확인할 수 없습니다'라고 답하세요.
답변의 각 핵심 주장 뒤에는 해당 근거의 대괄호 출처를 그대로 표시하세요.
"""
    response = client.responses.create(
        model=args.model,
        input=[
            {
                "role": "system",
                "content": "당신은 금융 문서 질의응답 도우미입니다.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    print("\nAnswer")
    print(response.output_text)


if __name__ == "__main__":
    main()
