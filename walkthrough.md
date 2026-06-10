# Self-Organising / Self-Healing RAG — Codebase Walkthrough

## Overview

A **FastAPI-based RAG system** with automatic failure detection and self-healing. It ingests documents into **Pinecone** (vector DB), answers queries via **Ollama** (local LLM), detects low-recall failures with 6 rule-based detectors, and automatically repairs them by re-chunking + re-embedding.

Metadata is logged to a local **SQLite** database, and a **Streamlit dashboard** provides visibility into query diagnostics, events, repairs, and evaluation history.

---

## Architecture

```
User ──► FastAPI (/api/v1) ──► Controllers ──► Services (Ollama + Pinecone)
                                    │
                              Logger (SQLite)
                                    │
                              Detector (6 rules)
                                    │
                              Repair Orchestrator
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              Re-chunk         Re-embed         Probe Score
           (semantic/llm/     (delete old +     (verify
            entropy)          insert new)       improvement)
                                    │
                              Streamlit Dashboard
```

---

## Project Structure

```
HPE_CPP/
├── main.py                          ← FastAPI entry + background self-healing loop
├── config.py                        ← Pydantic settings from .env
├── requirements.txt                 ← Python dependencies
├── .env                             ← Secrets (Pinecone API key, Ollama config)
│
├── api/
│   └── routes.py                    ← 9 REST endpoints
│
├── controllers/
│   ├── ingestion.py                 ← File upload → clean → chunk → embed → Pinecone
│   ├── retrieval.py                 ← Query → retrieve → LLM answer → log → detect
│   └── evaluation.py               ← Batch evaluation (ROUGE-L, semantic sim, context sim)
│
├── services/
│   └── llm_factory.py              ← Singleton factories: Ollama LLM, embeddings, Pinecone
│
├── db/
│   ├── models.py                    ← 4 SQLAlchemy tables: QueryLog, LowRecallEvent, RepairReport, EvalSnapshot
│   ├── session.py                   ← SQLite engine + init_db() + get_session()
│   ├── clear_db.py                  ← DB cleanup utility
│   ├── del_query.py                 ← Query deletion utility
│   └── migrate_add_chunks.py        ← Schema migration script
│
├── logger/
│   └── query_logger.py              ← log_query() + update_log_eval_metrics()
│
├── detector/
│   └── detectors.py                 ← 6 detection rules + run_detectors()
│
├── repair/
│   ├── orchestrator.py              ← Full repair loop: load → rechunk → replace → probe
│   ├── chunker.py                   ← 3 rechunking strategies (semantic, LLM, entropy)
│   └── reembedder.py                ← Safe partial re-embedding (delete specific IDs only)
│
├── auto_chunker/
│   ├── pipeline.py                  ← Orchestrates: segment → adaptive resize → final chunks
│   ├── semantic_chunker.py          ← Embedding-based topic boundary detection
│   ├── cluster_chunker.py           ← Agglomerative clustering on sentence embeddings
│   └── adaptive_chunker.py          ← Merge small chunks / split large ones (200–2000 tokens)
│
├── auto_indexer/
│   └── engine.py                    ← Staleness detection, partial re-embedding, consistency checks
│
├── dashboard/
│   └── app.py                       ← 5-tab Streamlit diagnostic UI
│
└── dataset/
    ├── contexts.docx                ← Main document corpus
    └── test.docx                    ← Test document
```

---

## Module Details

### 1. Entry Point — [main.py](file:///d:/Anantha/Academic/SEM%206/Xtra/HPE_CPP/main.py)

- Creates the FastAPI app, includes the router at `/api/v1`
- On startup: calls `init_db()` to create SQLite tables
- Launches a **background autonomous maintenance loop** (unless `ENV=evaluation`):
  - Polls every **10 seconds** for unresolved `LowRecallEvent` entries
  - Uses a **Multi-Strategy Waterfall**: `semantic → entropy → llm`
  - Each event gets up to **3 attempts** (one per strategy)
  - After 3 failures, the event is marked `unfixable = True`

### 2. Configuration — [config.py](file:///d:/Anantha/Academic/SEM%206/Xtra/HPE_CPP/config.py)

Pydantic `BaseSettings` loading from `.env`:
- **Pinecone**: `pinecone_api_key`, `pinecone_index_name`, `pinecone_namespace`
- **Ollama**: `embedding_model_name` (default: `mxbai-embed-large`), `ollama_base_url`
- **LLM**: `llm_provider` (default: `ollama`), `llm_model_name` (default: `mistral`)
- **Gemini**: `gemini_api_key` (for evaluation)

### 3. API Routes — [routes.py](file:///d:/Anantha/Academic/SEM%206/Xtra/HPE_CPP/api/routes.py)

| Method | Endpoint | Handler |
|--------|----------|---------|
| `POST` | `/ingest` | Upload file (PDF/DOCX/TXT) + optional namespace |
| `POST` | `/query` | RAG query → answer + contexts + scores + log_id |
| `POST` | `/evaluate` | Single-question evaluation |
| `POST` | `/evaluate-local` | Upload JSON dataset for batch evaluation |
| `POST` | `/auto-chunk` | Run auto-chunker pipeline, optionally ingest |
| `GET`  | `/logs` | Recent query logs |
| `GET`  | `/events` | Low-recall events (filterable) |
| `POST` | `/repair/{event_id}` | Manual repair trigger |
| `GET`  | `/index/health` | Pinecone index consistency check |
| `GET`  | `/index/staleness` | Detect stale embeddings |
| `POST` | `/index/refresh` | Full auto-indexer pipeline |

### 4. Controllers

#### [ingestion.py](file:///d:/Anantha/Academic/SEM%206/Xtra/HPE_CPP/controllers/ingestion.py)
1. Receives uploaded file → saves to temp
2. Loads with LangChain loaders (`PyPDFLoader`, `Docx2txtLoader`, `TextLoader`)
3. **Cleans text** — collapses whitespace, removes `\n`, `\t`
4. **Chunks** — `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)`
5. **Embeds + uploads** to Pinecone in batches of 50

#### [retrieval.py](file:///d:/Anantha/Academic/SEM%206/Xtra/HPE_CPP/controllers/retrieval.py)
1. Similarity search on Pinecone (`k=5`) — gets docs + cosine scores
2. Builds a LangChain `retrieval_chain` with `ChatPromptTemplate`
3. Generates answer via Ollama LLM
4. Calls `log_query()` → persists to SQLite
5. Calls `run_detectors(log_id)` → silent failure detection

> [!NOTE]
> Uses `langchain_classic.chains` — an older LangChain import path. This is the `create_retrieval_chain` / `create_stuff_documents_chain` pattern.

#### [evaluation.py](file:///d:/Anantha/Academic/SEM%206/Xtra/HPE_CPP/controllers/evaluation.py)
- **Metrics**: ROUGE-L (LCS-based), Semantic Similarity (cosine of embeddings), Context↔Query Similarity, Context↔Ground Truth Similarity
- `process_local_evaluation()`: Reads a JSON file with `qun`/`ans` fields, runs RAG pipeline on each, computes metrics, saves CSV + DB snapshot

### 5. Services — [llm_factory.py](file:///d:/Anantha/Academic/SEM%206/Xtra/HPE_CPP/services/llm_factory.py)

Cached singleton factories:
- `get_embeddings()` → `OllamaEmbeddings(model="mxbai-embed-large")` — 1024 dims
- `get_llm()` → `ChatOllama(model="mistral", temperature=0.2)`
- `get_pinecone_index()` → Pinecone `Index` object
- `get_vector_store(namespace)` → `PineconeVectorStore`

### 6. Database — [models.py](file:///d:/Anantha/Academic/SEM%206/Xtra/HPE_CPP/db/models.py) + [session.py](file:///d:/Anantha/Academic/SEM%206/Xtra/HPE_CPP/db/session.py)

**SQLite** at `db/autorag.db`. Four tables:

| Table | Purpose |
|-------|---------|
| `autorag_query_log` | Every query: text, chunk_ids, scores, LLM response, latency, flagged status |
| `autorag_low_recall_events` | Detector-fired events: severity, triggered detectors, resolved/unfixable flags, attempt count |
| `autorag_repair_reports` | Repair outcomes: strategy, chunks before/after, score before/after, duration |
| `autorag_eval_snapshots` | Evaluation run summaries: namespace, model config, average metrics |

### 7. Logger — [query_logger.py](file:///d:/Anantha/Academic/SEM%206/Xtra/HPE_CPP/logger/query_logger.py)

- `log_query()` — persists query + scores + answer + latency + chunk metadata → returns `log_id`
- `update_log_eval_metrics()` — backfills `answer_sem_sim`, `ctx_q_sim`, and `retrieved_chunks` after evaluation

### 8. Detector — [detectors.py](file:///d:/Anantha/Academic/SEM%206/Xtra/HPE_CPP/detector/detectors.py)

6 independent, non-blocking rules:

| # | Rule | What It Checks | Threshold |
|---|------|----------------|-----------|
| 1 | `low_top_score` | Top-1 retrieval score below threshold | < 0.45 |
| 2 | `score_drop` | Gap between rank-1 and rank-K | > 0.3 |
| 3 | `llm_uncertainty` | Hedging phrases in LLM response | keyword list |
| 4 | `semantic_mismatch` | Retrieved chunks semantically fragmented | mean pairwise sim < 0.55 |
| 5 | `evidence_mismatch` | LLM answer doesn't match evidence | answer↔evidence sim < 0.50 |
| 6 | `user_frustration` | Similar query within 300s | cosine sim ≥ 0.85 |

**Severity mapping**: 1 trigger = LOW, 2 = MEDIUM, 3+ = HIGH

### 9. Repair Pipeline

#### [orchestrator.py](file:///d:/Anantha/Academic/SEM%206/Xtra/HPE_CPP/repair/orchestrator.py)
1. Loads the `LowRecallEvent` + its `QueryLog`
2. Queries Pinecone directly to get **specific vector IDs** of failing chunks
3. Calls the chosen rechunk strategy
4. Calls `reembed()` to **safely replace only those chunks** (no full-document wipe)
5. Probes the score — marks resolved if improvement ≥ 5%
6. Persists a `RepairReport`

#### [chunker.py](file:///d:/Anantha/Academic/SEM%206/Xtra/HPE_CPP/repair/chunker.py)
- **semantic**: Smaller chunks (250 chars, 80 overlap) — best default
- **llm**: Asks Mistral to identify topic boundaries in the text
- **entropy**: Splits at sentences with high vocabulary novelty (>60% new words)

#### [reembedder.py](file:///d:/Anantha/Academic/SEM%206/Xtra/HPE_CPP/repair/reembedder.py)
- Deletes **only specific vector IDs** via `index.delete(ids=...)`
- Inserts new chunks via `vs.add_documents()`
- Safety: never deletes by source filter (prevents accidental full-doc wipeout)

### 10. Auto-Chunker — [auto_chunker/](file:///d:/Anantha/Academic/SEM%206/Xtra/HPE_CPP/auto_chunker)

A smart chunking pipeline exposed via `/api/v1/auto-chunk`:

1. **Semantic segmentation** — embeds sentences, finds boundaries where consecutive cosine similarity drops below threshold (0.65)
2. **Clustering** — agglomerative clustering on sentence embeddings, groups similar sentences into chunks
3. **Adaptive resize** — post-processor that merges chunks < 200 tokens and splits chunks > 2000 tokens at sentence boundaries

### 11. Auto-Indexer — [engine.py](file:///d:/Anantha/Academic/SEM%206/Xtra/HPE_CPP/auto_indexer/engine.py)

`AutoIndexer` class with 4 capabilities:
1. **Staleness detection** — samples vectors, re-embeds text, compares stored vs fresh embedding (drift threshold: 0.95)
2. **Partial re-embedding** — upserts only stale vectors in batches of 50
3. **Index refresh** — upserts new docs + deletes orphaned vectors (empty/short text)
4. **Consistency check** — vector count, dimension check, metadata integrity %, score distribution

### 12. Dashboard — [app.py](file:///d:/Anantha/Academic/SEM%206/Xtra/HPE_CPP/dashboard/app.py)

Streamlit app with 5 tabs: Overview, Query Diagnostics, Low-Recall Events, Repair History, Eval History. Communicates with the FastAPI backend via REST calls.

---

## Key Data Flows

### Ingestion Flow
```
File Upload → PyPDF/Docx2txt/Text Loader → Clean (regex whitespace collapse)
→ RecursiveCharacterTextSplitter(500, 100) → Batch embed (50/batch) → Pinecone upsert
```

### Query Flow
```
User Query → Pinecone similarity_search(k=5) → Get docs + scores
→ LangChain retrieval_chain (stuff docs into prompt) → Ollama Mistral generates answer
→ log_query() to SQLite → run_detectors() (6 rules, silent)
→ If triggered: create LowRecallEvent(severity, detectors)
→ Return {answer, contexts, scores, log_id}
```

### Self-Healing Flow
```
Background loop (every 10s) picks unresolved events
→ Waterfall: attempt 1=semantic, 2=entropy, 3=llm
→ orchestrator.handle_event():
    → Get failing chunk IDs from Pinecone
    → Rechunk text with chosen strategy
    → Delete old IDs + insert new chunks
    → Probe: re-query, compare scores
    → If score_after > score_before + 0.05 → RESOLVED
    → Else: next cycle tries next strategy
→ After 3 failures → unfixable = True
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **API** | FastAPI + Uvicorn |
| **LLM** | Ollama (Mistral) |
| **Embeddings** | Ollama (mxbai-embed-large, 1024 dims) |
| **Vector DB** | Pinecone |
| **Metadata DB** | SQLite via SQLAlchemy |
| **RAG Framework** | LangChain (langchain-classic chains) |
| **Dashboard** | Streamlit |
| **Evaluation** | Custom (ROUGE-L, cosine similarity) |
| **ML** | scikit-learn (clustering), NumPy |

---

## Configuration

Environment variables (`.env`):
```
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=...
PINECONE_NAMESPACE=...
EMBEDDING_MODEL_NAME=mxbai-embed-large
OLLAMA_BASE_URL=http://localhost:11434
LLM_PROVIDER=ollama
LLM_MODEL_NAME=mistral:latest
```

---

## Notable Design Decisions

1. **Safe repairs** — Only specific failing chunk IDs are deleted/replaced, never full documents
2. **Multi-strategy waterfall** — Escalates through 3 repair strategies before giving up
3. **Non-blocking detection** — Detectors never raise exceptions or block the API response
4. **Dual DB architecture** — Pinecone for vectors, SQLite for metadata/logs/events
5. **Evaluation mode** — `ENV=evaluation` disables the background self-healing loop to get clean baseline metrics
