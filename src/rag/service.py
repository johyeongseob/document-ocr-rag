"""Reusable retrieval and answer-generation service for CLI and web clients."""

from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI

from .utils import cosine_similarity


def answer_question(
    question: str,
    index_path: Path = Path("outputs/rag_index.json"),
    top_k: int = 3,
    model: str = "gpt-5-mini",
) -> dict:
    """Retrieve relevant OCR chunks and generate a cited Korean answer."""
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set the OPENAI_API_KEY environment variable first.")
    if not index_path.is_file():
        raise FileNotFoundError(f"RAG index not found: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    client = OpenAI()
    query_embedding = client.embeddings.create(
        model=index["embedding_model"], input=question
    ).data[0].embedding

    ranked = sorted(
        (
            (cosine_similarity(query_embedding, chunk["embedding"]), chunk)
            for chunk in index["chunks"]
        ),
        key=lambda item: item[0],
        reverse=True,
    )[:top_k]

    contexts = []
    evidence_parts = []
    for score, chunk in ranked:
        citation = f"[{chunk['source']}#{chunk['id']}]"
        evidence_parts.append(f"{citation}\n{chunk['text']}")
        contexts.append(
            {
                "citation": citation,
                "source": chunk["source"],
                "chunk_id": chunk["id"],
                "similarity": round(score, 4),
                "text": chunk["text"],
            }
        )

    evidence = "\n\n".join(evidence_parts)
    prompt = f"""질문:
{question}

검색된 OCR 근거:
{evidence}

검색된 근거에 직접 명시된 내용만 사용해 자연스러운 한국어로 간결하게 답하세요.
서로 다른 조건이나 문장을 임의로 연결하지 마세요.
근거가 부족하면 추측하지 말고 '제공된 문서에서 확인할 수 없습니다'라고 답하세요.
각 핵심 주장 뒤에는 해당 근거의 대괄호 출처를 그대로 표시하세요.
"""
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "당신은 검색된 근거만 사용하는 금융 문서 질의응답 도우미입니다.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return {"answer": response.output_text, "contexts": contexts}

