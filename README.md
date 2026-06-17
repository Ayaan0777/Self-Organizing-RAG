# Self-Organising RAG

A self-healing Retrieval-Augmented Generation system that detects, diagnoses,
and repairs its own retrieval failures — no human in the loop.

The system answers user questions over an ingested document corpus, runs 5
detection rules over every answer, and when failures cross a threshold it
fires a **single-pass 4-strategy repair cascade**. Successful strategies can
be **promoted** into the main query pipeline.

---

## What it does

```
User query → Gate → Retrieve → LLM answer → Log → Detect
                                            └→ (background) Enrich + Detect
                                                              ↓
                                                    LowRecallEvent
                                                              ↓
                                        Auto-worker polls every 5s
                                                              ↓
                                       30%+ flagged AND ≥5 pending?
                                                              ↓
                                                       Repair cascade
                                                              ↓
                                    S1 dynamic K → S2 chunk size →
                                    S3 combined → S4 alternate LLM
                                                              ↓
                                    First strategy that improves wins.
                                    All four fail → mark unfixable.
```

When S1 (dynamic K) resolves 5 events, it's promoted to the main pipeline
and future queries get category-aware retrieval (K=2–10 instead of fixed K=5).

---

## Tech stack

| Layer | Technology |
|---|---|
| HTTP API | FastAPI |
| Primary LLM | Ollama — `mistral` (7B) |
| Fallback LLM | Ollama — `gemma3:27b` (cascade S4) |
| Embeddings | Ollama — `mxbai-embed-large` (1024-dim) |
| Vector store | Pinecone |
| Persistence | SQLite via SQLAlchemy |
| Dashboard | Streamlit |
| Glue | LangChain primitives |

---

## Quick start

Three processes share one SQLite DB and one Pinecone index.

```powershell
# 1. Install
pip install -r requirements.txt

# 2. Ollama setup
ollama serve
ollama pull mistral
ollama pull gemma3:27b
ollama pull mxbai-embed-large

# 3. Configure .env (see env_format.txt for required keys)
#    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE, etc.

# 4. Run all three processes in separate terminals
uvicorn main:app --reload                    # Terminal 1 — API
python auto_worker.py                         # Terminal 2 — repair daemon
streamlit run dashboard/app.py                # Terminal 3 — dashboard
```

Dashboard at <http://localhost:8501>, API at <http://localhost:8000/api/v1>.

To reset everything between test runs:

```powershell
python db/clear_db.py --all --confirm
```

This wipes Pinecone vectors AND all 9 SQLite tables, including strategy
counters and runtime flags so promotion state is reset too.

---

## Repository layout

```
HPE_CPP/
├── main.py                   # FastAPI app entry
├── auto_worker.py            # Cascade trigger daemon
├── config.py                 # Pydantic settings
├── api/routes.py             # All HTTP endpoints
├── controllers/
│   ├── retrieval.py          # answer_query, gate, post-process thread
│   ├── ingestion.py          # Document → chunks → Pinecone
│   ├── evaluation.py         # Batch eval via /evaluate-local
│   ├── metrics.py            # precision / sufficiency / hallucination
│   └── gt_lookup.py          # Dataset GT enrichment
├── detector/
│   ├── detectors.py          # 5 detection rules
│   └── decision_engine.py    # diagnose() + STRATEGY_CONFIGS
├── repair/
│   ├── cascade.py            # Ordered cascade + counters + promotion
│   ├── orchestrator.py       # handle_event primitive + _probe_metrics
│   ├── chunker.py            # rechunk_semantic / llm / entropy
│   └── reembedder.py         # Snapshot + rollback
├── organiser/retrieval_gate.py
├── add_chunks/pipeline.py
├── auto_indexer/engine.py    # Staleness detection
├── services/llm_factory.py   # Cached singletons
├── logger/query_logger.py
├── db/
│   ├── models.py             # 9 SQLAlchemy tables
│   ├── session.py
│   ├── clear_db.py
│   └── del_query.py
├── dashboard/app.py          # Streamlit, 9 pages
├── dataset/long_ans.json     # GT dataset (default)
├── unused/                   # Legacy 384-dim scripts — do not run
└── MD_files/                 # All design + audit documentation
    ├── explanation.md        # Full system explanation
    ├── claude.md             # Cascade implementation plan
    ├── bugs.md               # Bug audit + resolution log
    ├── walkthrough.md        # Cascade implementation walkthrough
    ├── notes_compare.md      # Spec-vs-built coverage
    ├── instruction.md        # Variable chunk size spec
    ├── stages_explained.md   # Stage 1-4 reference
    └── implementation_plan.md
```

---

## Key features

- **5 detection rules** with K-adaptive thresholds — rule 2 uses max
  adjacent gap (K-invariant), rule 4 uses `0.65 × top1_score` (relative)
- **4-strategy single-pass cascade** with per-strategy K-matched baselines
  and cascade-owned rollback
- **Snapshot + rollback safety** — every Pinecone modification is reversible
  via `ChunkSnapshot`
- **One-way promotion** — S1 (dynamic K) gets elevated to the main pipeline
  after 5 successes
- **GT-backed inline enrichment** — user queries that match
  `dataset/long_ans.json` automatically get precision/recall/sufficiency
  computed and persisted; no separate `/evaluate-local` run needed
- **Retrieval gate** — chitchat ("hi", "thanks") bypasses Pinecone entirely
- **Background-thread post-processing** — answer returns in ~3s; detection
  and GT enrichment happen async
- **Streamlit dashboard** with 9 pages: Overview, Ingest, Ask Query,
  Add Chunks, Query Diagnostics, Flagged Events, Eval History, Pipeline
  Config, Adaptation Log

---

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/ingest` | Upload + ingest PDF/DOCX/TXT |
| `POST` | `/api/v1/query` | Ask a question |
| `POST` | `/api/v1/add-chunks` | Ad-hoc text chunking |
| `POST` | `/api/v1/evaluate-local` | Batch evaluate a JSON dataset |
| `POST` | `/api/v1/repair/{event_id}` | Manually trigger the cascade |
| `GET` | `/api/v1/logs` | Recent query logs |
| `GET` | `/api/v1/events` | LowRecallEvents |
| `GET` | `/api/v1/repair-report/{event_id}` | Repair report + diagnosis |
| `GET` | `/api/v1/strategy-counters` | Strategy success counts |
| `GET` | `/api/v1/runtime-flags` | Promotion flags |
| `GET` | `/api/v1/adaptation-log` | Cascade audit trail |
| `GET` | `/api/v1/index/health` | Pinecone consistency check |
| `GET` | `/api/v1/index/staleness` | Embedding drift detection |

---

## Performance

Typical user-visible response time: **~3.0s** on a CPU Ollama setup.

| Stage | Cost |
|---|---|
| Embed query + Pinecone search | ~280ms |
| Mistral inference | ~2700ms (60% of total) |
| `log_query` | ~20ms |
| Detection + GT enrichment | (background, ~900ms hidden) |

Optimized down from ~4.4s through four changes: dropped duplicate Pinecone
search, batched detector embeddings (`embed_documents` instead of N
`embed_query` calls), moved post-processing to a daemon thread, and reused
the query embedding. Further gains require a smaller LLM (e.g. `llama3.2:3b`,
`gemma3:4b`) or GPU acceleration — mistral is the floor.

---

## Documentation map

All design notes and audit logs are under `MD_files/`. Open them directly
or follow the links below.

| File | Purpose |
|---|---|
| [MD_files/explanation.md](MD_files/explanation.md) | Full system explanation — architecture, every module, mermaid diagrams, end-to-end flow |
| [MD_files/claude.md](MD_files/claude.md) | Original cascade implementation plan (historical reference) |
| [MD_files/bugs.md](MD_files/bugs.md) | Bug audit log — 23 bugs categorized, 20 fixed, 3 explicitly ignored |
| [MD_files/notes_compare.md](MD_files/notes_compare.md) | Coverage of the original deep-dive spec vs. what was built |
| [MD_files/walkthrough.md](MD_files/walkthrough.md) | Cascade implementation walkthrough |
| [MD_files/instruction.md](MD_files/instruction.md) | Variable chunk size ingestion spec |
| [MD_files/stages_explained.md](MD_files/stages_explained.md) | Stage 1-4 (DETECT / MEASURE / DECIDE / ACT) reference |

---

## Project status

All 7 core components from the original deep-dive spec are present. All 6
monthly milestones complete. See [notes_compare.md](notes_compare.md) for
the spec-vs-built breakdown. The architecture goes beyond the brief in
several places — ordered cascade with promotion, GT-backed inline metrics,
K-adaptive detection, snapshot rollback — that weren't in the original
plan.

For the prioritized list of remaining cleanup items, see [bugs.md](bugs.md).
