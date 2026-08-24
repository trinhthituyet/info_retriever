# info-retriever

Agentic RAG over your own contract documents — rental agreements, employment
contracts, insurance policies. Add documents, ask questions in natural language,
get answers with exact page citations.

Claude does the reading and reasoning. **Embeddings run locally**, so your salary,
address and policy numbers are never sent to a second vendor.

## How it works

```
      add                                        ask
       │                                           │
   ┌───▼────┐  no text layer?  ┌──────────┐   ┌────▼──────────────────────┐
   │ loader │ ───────────────► │ Claude   │   │ agent (Claude + 3 tools)  │
   │ pdf    │                  │ vision   │   │  query_documents  (SQL)   │
   │ docx   │ ◄─── text ────── │ (OCR)    │   │  search_chunks (hybrid)   │
   │ image  │                  └──────────┘   │  read_document (full text)│
   │ txt    │                                 └────┬──────────────────────┘
   └───┬────┘                                      │ documents it opened
       │                                      ┌────▼──────────────────────┐
   ┌───▼──────────────┐  ┌────────────────┐   │ citation pass             │
   │ Claude:          │  │ chunk (clause- │   │ re-send those documents   │
   │ classify +       │  │ aware) + embed │   │ with citations enabled    │
   │ extract fields   │  │ locally        │   └────┬──────────────────────┘
   └───┬──────────────┘  └────────┬───────┘        │
       └──────────┬───────────────┘           answer + page spans
                  ▼
      SQLite: documents + chunks + chunk_vec + chunk_fts
```

### The two design decisions worth knowing

**1. Chunk RAG is the fallback, not the main path.** With tens or hundreds of
documents, a compact catalogue (one line per document) fits in the system prompt
behind a cache breakpoint. The agent reads that, picks the relevant document, and
reads it *whole*. Contract clauses depend on definitions and figures stated
elsewhere in the same document — stitching excerpts together is how contract
answers go wrong. `search_chunks` exists to *find* the right document when the
catalogue is ambiguous, not to be the primary retrieval path.

**2. Structured extraction beats semantic search for the questions people
actually ask.** "When does my lease expire", "what's my notice period", "which
policies renew this quarter" are SQL queries over typed fields, not vector
searches. Ingestion extracts those fields into a per-document-type schema
(`schemas.py`), and `query_documents` filters them directly.

## Setup

```bash
uv venv --python 3.12
uv pip install -e .
cp .env.example .env         # then put your ANTHROPIC_API_KEY in it
info-retriever               # http://127.0.0.1:8000
```

The first upload downloads the embedding model (~2.2 GB for the default
`BAAI/bge-m3`) and caches it in `~/.cache/huggingface`. To trade quality for
speed and disk, set in `.env`:

```bash
EMBED_MODEL=intfloat/multilingual-e5-small
EMBED_DIM=384
EMBED_QUERY_PREFIX=query:•             # e5 models require these; use a space, not •
EMBED_PASSAGE_PREFIX=passage:•
```

Changing the model means changing `EMBED_DIM`, and the vector table's dimension is
fixed at creation — delete `data/documents.db` and re-ingest.

## Using it

```bash
info-retriever                       # 127.0.0.1:8000
info-retriever --port 3000 --reload
```

Drop files on the left to ingest; ask on the right. Both are slow operations
(vision transcription, two Claude passes), so both stream over SSE rather than
hanging on a spinner: you see each ingest stage per file, and each tool the agent
calls, then the cited answer streams in token by token. Click any document to see
its extracted fields, download the original, or remove it from the index.

**Verify & cite** (on by default) runs the second Claude pass, which re-answers from
the original documents and returns page references. Turning it off roughly halves
cost and latency but gives no provenance.

**Inspect retrieval** runs the hybrid search alone, with no model in the loop, and
shows the ranked passages. This is the first thing to try when an answer looks
wrong — it separates a retrieval problem from a reasoning problem.

The frontend is one HTML file plus vanilla JS and CSS served by FastAPI — no npm,
no build step, no bundler. Everything document- or model-derived is written with
`textContent`, never `innerHTML`: titles and cited text come from files we did not
author, so treating them as markup would be an injection path.

It binds to `127.0.0.1` and has **no authentication**. Do not expose it to a
network — the whole corpus is readable by anyone who can reach the port.

### API

| Endpoint | Purpose |
|---|---|
| `GET /api/stats` | Counts and configured models. |
| `GET /api/documents` | Catalogue. |
| `GET /api/documents/{id}` | Extracted structured fields. |
| `GET /api/documents/{id}/file` | Download the stored original. |
| `DELETE /api/documents/{id}` | Remove from the index. |
| `POST /api/uploads` | Multipart upload → `{job_id}`. |
| `GET /api/uploads/{job_id}/events` | SSE: `file_start`, `progress`, `file_done`, `file_skipped`, `file_failed`, `summary`. |
| `GET /api/ask?q=&cite=` | SSE: `stage`, `tool`, `draft`, `delta`, `answer`, `failure`. |
| `GET /api/search?q=&limit=&doc_type=` | Raw hybrid retrieval, no LLM. |

Interactive docs at `/api/docs`. Ingestion is serialised behind a lock — SQLite
tolerates one writer, and the embedding model is not worth loading twice.

The API is plain JSON and SSE with no auth, so `curl` works for scripting:

```bash
curl -N 'http://127.0.0.1:8000/api/ask?q=What+is+my+notice+period%3F&cite=false'
curl 'http://127.0.0.1:8000/api/search?q=termination+notice&limit=5'
curl -F files=@lease.pdf http://127.0.0.1:8000/api/uploads
```

## Layout

| File | Role |
|---|---|
| `db.py` | **All** SQL. Schema, writes, structured queries, vector + keyword search. |
| `schemas.py` | Pydantic models = the structured-output schemas Claude extracts into. |
| `loaders.py` | File → Claude content blocks + text. Decides if OCR is needed. |
| `chunking.py` | Clause- and page-aware segmentation. |
| `embed.py` | Local embedding model (lazy-loaded; torch import is slow). |
| `extract.py` | Claude calls: transcribe, classify, extract fields. |
| `ingest.py` | Orchestrates the add pipeline. |
| `retrieval.py` | Hybrid search with Reciprocal Rank Fusion. |
| `tools.py` | The three agent tools. |
| `agent.py` | Agentic pass + citation pass. |
| `web.py` | FastAPI routes, SSE streaming, ingest job runner, entry point. |
| `static/` | The frontend: `index.html`, `app.js`, `style.css`. |

## Notes and known limits

- **Citations and structured outputs are mutually exclusive** in the API (a request
  with both returns 400). That is why answering is two calls: an agentic pass that
  uses tools, then a citation pass over the documents it opened. `--no-cite` skips
  the second.
- **Page-level citations need the original PDF.** For PDFs the citation pass
  re-sends the stored file, so you get page numbers. For DOCX/text it sends the
  extracted text, so you get character offsets. Images can't be cited at all —
  their transcription is cited as text instead.
- **Extraction reads text, not pixels** (except when transcribing a scan). If a
  table-heavy document extracts badly, the knob is in `extract.py`: pass
  `loaded.content_blocks` alongside the text so Claude sees the layout. Costs more
  image tokens.
- **The catalogue is cached, not free.** It sits behind a `cache_control`
  breakpoint, so repeated questions read it at ~0.1× input price. It is
  invalidated whenever you add a document — the first question after an ingest
  pays a cache write.
- **Not legal advice.** The agent is instructed to state what the documents say and
  flag ambiguity rather than resolve it. Treat output as a fast index into your own
  paperwork, not an opinion.

## Moving to Postgres + pgvector

SQLite is the right default here: no server, and the whole corpus is one file you
can encrypt and back up. Outgrow it — concurrent writers, a web frontend, more than
a few thousand documents — and only `db.py` changes:

| SQLite | Postgres |
|---|---|
| `chunk_vec` virtual table (`vec0`) | `embedding vector(1024)` column + HNSW index |
| `chunk_fts` virtual table (`fts5`) | `tsvector` generated column + GIN index |
| `extracted` TEXT holding JSON | `jsonb` + generated date columns |
| `bm25(chunk_fts)` | `ts_rank_cd(tsv, query)` |

The RRF fusion in `retrieval.py`, the tools, and the agent are all storage-agnostic
and carry over unchanged.

## Tests

```bash
.venv/bin/python -m pytest -q
```

No API key or model download required. `test_pipeline.py` covers chunking, date
normalisation and a storage round-trip with deterministic fake vectors;
`test_web.py` drives the real FastAPI routes and SSE framing with Claude and the
embedding model stubbed.
