# RAG

This module converts OCR predictions into a searchable index, retrieves the
chunks most relevant to a question, and asks GPT-5 mini to produce an answer
grounded in those chunks.

## Pipeline

```text
OCR prediction lines
        ↓
Source-aware character chunks
        ↓
text-embedding-3-small vectors
        ↓
Cosine-similarity Top-K retrieval
        ↓
GPT-5 mini answer with document evidence
```

The prototype reads OCR predictions from `outputs/evaluation_all.json`. It uses
`text-embedding-3-small` for retrieval embeddings and `gpt-5-mini` for grounded
answer generation.

## Model roles

`text-embedding-3-small` is an OpenAI embedding model. It converts each OCR text
chunk and the user's question into numerical vectors. The system compares these
vectors with cosine similarity and selects the most relevant chunks. It does
not write the answer.

`gpt-5-mini` is the generation model. It receives the question and retrieved
chunks, then writes the final answer using only that evidence.

```text
OCR chunk ─┐
           ├─ text-embedding-3-small → vector similarity → relevant chunks
Question ──┘                                                      ↓
                                                              gpt-5-mini → grounded answer
```

## API setup and billing

An OpenAI API account with available paid credits is required. New prepaid
accounts may require a minimum initial credit purchase, commonly USD 5. Check
the current billing terms and auto-recharge setting before running the example.

Set the API key only in the current PowerShell session and never commit it:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

## Build an index

Build an index from the first **10** OCR documents:

```powershell
python -m src.rag.build_index `
  --report .\outputs\evaluation_all.json `
  --limit 10 `
  --output .\outputs\rag_index.json
```

After an index has been created, questions reuse the stored chunk embeddings.
Rebuild it only when the OCR results, chunking configuration, embedding model,
or selected documents change. To index the full 180-document sample, use
`--limit 180`; the existing output file can be overwritten.

## Ask a question

```powershell
python -m src.rag.query `
  "개인신용정보 조회에 동의하지 않으면 어떤 불이익이 있나요?" `
```

Example questions:

1. `개인신용정보 조회에 동의하지 않으면 어떤 불이익이 있나요?`
2. `개인신용정보를 수집하고 이용하는 목적은 무엇인가요?`
3. `조회 동의의 효력기간이 끝난 뒤에는 어떤 목적으로 정보를 보유하나요?`
4. `개인신용정보를 조회하는 대상 기관은 어디인가요?`
5. `문서에서 조회하거나 수집하는 개인정보 항목에는 무엇이 있나요?`

The model is instructed to use only retrieved OCR evidence. If the evidence is
insufficient, it should state that rather than infer an unsupported answer.

## Web interface

After creating `outputs/rag_index.json` and setting `OPENAI_API_KEY`, start the
FastAPI application:

```powershell
python -m uvicorn web.app:app --reload
```

Open `http://127.0.0.1:8000` in a browser. The interface provides example
questions, a Top-K selector, a grounded answer, compact evidence citations,
source documents, similarity scores, and expandable OCR text. The API key stays
in the server's PowerShell environment and is never sent to the browser.

## Source files

| File | Role |
|---|---|
| `build_index.py` | Load OCR documents, create chunks and request embeddings |
| `query.py` | Command-line question-answering interface |
| `service.py` | Shared retrieval and generation service used by CLI and web |
| `utils.py` | Document loading, chunking, and cosine similarity utilities |

## Data security

OCR text and retrieved chunks can contain personal or sensitive information.
Do not send non-public documents to an external API without authorization and
appropriate de-identification. Generated indexes are stored under `outputs/`
and excluded from Git.
