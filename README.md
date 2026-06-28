# Self-Organising RAG

A Retrieval-Augmented Generation framework that **autonomously monitors, detects, diagnoses, and repairs** its own retrieval failures — without human intervention. Standard RAG pipelines silently degrade when chunk sizes are wrong, irrelevant content is retrieved, or the LLM hallucinates; they have no way to know something went wrong, let alone fix it. This system closes that loop with a 5-rule detection engine, a 4-strategy repair cascade with safe rollback, and a learning mechanism that promotes successful strategies into the main pipeline.

```
┌──────────┐    ┌───────────┐    ┌─────────┐    ┌────────────┐    ┌─────────┐
│   User   │───▶│ Retrieval │───▶│   LLM   │───▶│ Evaluation │───▶│ Deliver │
│  Query   │    │  Gate +   │    │ Answer  │    │  Engine    │    │Response │
└──────────┘    │ Pipeline  │    └─────────┘    └─────┬──────┘    └─────────┘
                └───────────┘                         │
                      ▲                          Good?│
                      │                    ┌─────────┴─────────┐
                      │                   YES                  NO
                      │                    │           ┌───────▼────────┐
                ┌─────┴──────┐             │           │  Self-Healing  │
                │ Adaptation │◀────────────┘           │ Repair Cascade │
                │ & Strategy │                         │  S1→S2→S3→S4  │
                │ Promotion  │◀────────────────────────┴────────────────┘
                └────────────┘
          Continuous Loop: Detect → Diagnose → Repair → Adapt → Improve
```

---

## Prerequisites

Before installation, ensure you have the following:

| Requirement | Notes |
|---|---|
| **Python 3.10+** | 3.11 recommended (used in the Docker image) |
| **Ollama** |  Download from [ollama.com](https://ollama.com). |
| **Pinecone account** | Free tier works. You need an API key and an index with **1024 dimensions** (to match mxbai-embed-large). |
| **Docker** *(optional)* | Only needed for containerised setup (Option B). |

### Ollama models

The default configuration uses these models:

| Model | Role | Size |
|---|---|---|
| `mxbai-embed-large` | Embeddings (1024 dims) | ~670 MB |
| `mistral` | Primary LLM | ~4.1 GB |
| `gemma3:27b` | Fallback LLM (repair Strategy 4) | ~17 GB, **requires ~20 GB RAM** |

> [!NOTE]
> **Using a low-end machine?** You can swap the primary or fallback LLM for a lighter model.
> Update `LLM_MODEL_NAME` or `FALLBACK_LLM_MODEL` in your `.env` and pull the model with `ollama pull <model_name>`.
>
> Some lighter options:
> - `gemma3:4b` (~3 GB)
> - `llama3.2:3b` (~2 GB)
> - `phi4-mini` (~2.5 GB)

---

## Features

- **Provides full observability** — a Streamlit dashboard with real-time query logs, flagged events, repair reports, evaluation trends, and pipeline configuration
- **Splits documents intelligently** — variable chunk sizing (500–1250 chars) with a semantic separator hierarchy preserves paragraph and sentence boundaries, with minimum-size enforcement to prevent fragment chunks
- **Supports incremental knowledge expansion** — add new content at any time through the add-chunks endpoint without re-ingesting existing documents
- **Understands query intent before searching** — a 2-tier retrieval gate (pattern matching + LLM classification) skips the vector database for greetings and chitchat, saving time and avoiding noise
- **Retrieves the right amount of context per question** — Dynamic K selection classifies queries into factual/complex/cross-section and picks optimal chunk count (2–10)
- **Detects bad answers automatically** — 5 independent detection rules (low score, score drop, LLM uncertainty, semantic mismatch, evidence mismatch) run on a background thread after every query
- **Measures quality with 7 metrics** — ROUGE-L, semantic similarity, context relevance, retrieval precision@K, context sufficiency, and hallucination rate — all embedding-based (no extra LLM calls)
- **Diagnoses root causes** — a decision engine maps failure patterns to specific chunk configurations (e.g., high hallucination → tighter chunks, insufficient context → larger chunks)
- **Repairs itself without human intervention** — a 4-strategy ordered cascade (Dynamic K → Rechunking → Combined → Alternate LLM) with snapshot-based rollback ensures data is never permanently lost
- **Learns from its own repairs** — when Dynamic K accumulates 5 successful fixes, it gets permanently promoted to the main query pipeline
- **Keeps embeddings fresh** — an auto-indexer detects stale/drifted embeddings by comparing stored vs. freshly computed vectors, and refreshes them in-place via upsert


---

## Installation

### Option A — Local machine

**1. Clone the repository**

```bash
git clone https://github.com/Ayaan0777/Self-Oragnising-RAG.git
cd Self-Oragnising-RAG
```

**2. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure environment variables**

```bash
cp env_format.txt .env
```

Edit `.env` with your values — see [Configuration](#configuration) below.

**4. Install Ollama and pull models**

Download Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama pull mxbai-embed-large
ollama pull mistral
ollama pull gemma3:27b          # skip if on low-end hardware — see Prerequisites
```

**5. Start all three processes** (each in a separate terminal)

```bash
# Terminal 1 — API server
uvicorn main:app --reload

# Terminal 2 — Self-healing repair daemon
python auto_worker.py

# Terminal 3 — Monitoring dashboard
streamlit run dashboard/app.py
```

The API will be available at `http://localhost:8000/docs` (Swagger UI) and the dashboard at `http://localhost:8501`.

---

### Option B — Docker

**1. Clone and configure**

```bash
git clone https://github.com/Ayaan0777/Self-Oragnising-RAG.git
cd Self-Oragnising-RAG
cp env_format.txt .env
```

Edit `.env` — set `OLLAMA_BASE_URL=http://host.docker.internal:11434`.

**2. Start Ollama on your host machine**

Ollama must be running on the host — the containers connect to it via `host.docker.internal`:

```bash
ollama serve
```

**3. Build and run**

```bash
docker-compose up -d
```

This starts three containers:
- `rag-api` — FastAPI server on port **8000**
- `rag-worker` — auto_worker.py repair daemon
- `rag-dashboard` — Streamlit dashboard on port **8501**

All three share a persistent `rag_db` volume for the SQLite database.

> [!NOTE]
> **Linux users:** `host.docker.internal` may not resolve by default. Add `--add-host=host.docker.internal:host-gateway` to your Docker run command, or add it under `extra_hosts` in `docker-compose.yml`.

---

## First Run / Quick Start

Once the system is running, here's how to get started:

### 1. Ingest a document

Open the dashboard at `http://localhost:8501` and navigate to the **Ingest Document** page. Upload a PDF, DOCX, or TXT file and select your namespace.

> **Alternatively**, use Swagger UI at `http://localhost:8000/docs` → `POST /api/v1/ingest`, or via curl:
> ```bash
> curl -X POST http://localhost:8000/api/v1/ingest \
>   -F "file=@your_document.pdf" \
>   -F "namespace=my-namespace"
> ```

### 2. Ask a question

Go to the **Ask Query** page in the dashboard, type your question, and hit submit. The response shows the answer, retrieved chunks, similarity scores, and gate decision.

> **Alternatively**, use Swagger UI → `POST /api/v1/query`, or via curl:
> ```bash
> curl -X POST http://localhost:8000/api/v1/query \
>   -H "Content-Type: application/json" \
>   -d '{"query": "What is binary search?", "namespace": "my-namespace"}'
> ```

### 3. Check system health

The dashboard homepage shows real-time stats — total queries, healthy count, flagged events, and resolved repairs. The dashboard has the following pages: **Overview**, **Ingest Document**, **Ask Query**, **Query Logs**, **Flagged Events**, **Repair Reports**, **Evaluation**, **Eval History**, and **Pipeline Config**.

> **Alternatively**, use the API directly:
> - `GET /api/v1/logs` — recent query logs
> - `GET /api/v1/events` — flagged events
> - `GET /api/v1/repair-report/{event_id}` — repair details for a specific event



---

## Configuration

A `.env` file is required to run the project. The complete template with all variables and default values is provided in **`env_format.txt`** in the repository root.

```bash
cp env_format.txt .env
```

**You must fill in these values:**

| Variable | Description |
|---|---|
| `PINECONE_API_KEY` | Your Pinecone API key |
| `PINECONE_INDEX_NAME` | Name of your Pinecone index (must be 1024 dimensions) |
| `PINECONE_NAMESPACE` | Namespace to use within the index |

**You may want to change these:**

| Variable | Default | When to change |
|---|---|---|
| `LLM_MODEL_NAME` | `mistral:latest` | If you prefer a different primary LLM |
| `FALLBACK_LLM_MODEL` | `gemma3:27b` | **Must change** if your machine has < 20 GB RAM — use `gemma3:4b` or `llama3.2:3b` |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Use `http://localhost:11434` for local setup without Docker |

All other variables (detection thresholds, auto-worker intervals, metric thresholds, dataset path) have sensible defaults and don't need to be changed for a basic run.

---

## API Reference

All endpoints are prefixed with `/api/v1`. Swagger UI available at `http://localhost:8000/docs`.

### Core RAG

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/ingest` | Upload and chunk a document (PDF, DOCX, or TXT) |
| `POST` | `/query` | Ask a question — full RAG pipeline with retrieval gate |
| `POST` | `/add-chunks` | Chunk raw text; optionally ingest to Pinecone |
| `POST` | `/evaluate-local` | Batch evaluation from a JSON dataset upload |

### Monitoring & Logs

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/logs` | Recent query logs with scores, latency, and flags |
| `GET` | `/events` | Flagged LowRecallEvents (filterable by unresolved) |

### Repair & Self-Healing

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/repair/{event_id}` | Manually trigger repair cascade for one event |
| `GET` | `/repair-report/{event_id}` | Repair report + root-cause diagnosis |
| `GET` | `/adaptation-log` | Full provenance trail of all adaptation cycles |
| `GET` | `/strategy-counters` | Success counts per cascade strategy (S1–S4) |
| `GET` | `/runtime-flags` | Runtime flags (e.g., `dynamic_k_promoted`) |

### Pipeline Configuration

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/pipeline-config` | Current active chunk configuration |
| `POST` | `/pipeline-config` | Manually override chunk config for testing |

### Auto-Indexer

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/index/health` | Consistency check — vectors, dimensions, metadata |
| `GET` | `/index/staleness` | Detect stale/drifted embeddings |
| `POST` | `/index/refresh` | Full pipeline: detect stale → re-embed → verify |

### Evaluation History

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/eval-history` | Evaluation snapshots for dashboard trend charts |

---

## Project Structure

```
Self-Oragnising-RAG/
├── api/                  # FastAPI route definitions (19 endpoints)
├── controllers/          # Core logic — ingestion, retrieval, evaluation, metrics, GT lookup
├── organiser/            # Retrieval gate (2-tier query classification)
├── detector/             # 5 detection rules + decision engine for root-cause diagnosis
├── repair/               # 4-strategy cascade, orchestrator, chunker, reembedder with rollback
├── auto_indexer/         # Stale embedding detection and in-place refresh
├── add_chunks/           # Incremental knowledge addition pipeline
├── services/             # LLM, embedding, and Pinecone singleton factories
├── logger/               # Query logging and metric persistence
├── db/                   # SQLAlchemy models (9 tables) and session management
├── dashboard/            # Streamlit monitoring dashboard
├── dataset/              # Ground-truth Q&A dataset (long_ans.json)
├── main.py               # FastAPI application entry point
├── auto_worker.py        # Background daemon for batch-driven self-healing
├── config.py             # Pydantic settings (all configurable via .env)
├── docker-compose.yml    # 3-container deployment (api + worker + dashboard)
├── Dockerfile            # Python 3.11-slim based image
├── requirements.txt      # Python dependencies
└── env_format.txt        # .env template with all variables and defaults
```

---

## Performance

Typical latency per operation (measured on a system with **16 GB RAM, AMD Ryzen 7, RTX 3060, running Ollama locally**):

| Operation | Latency | Notes |
|---|---|---|
| Query (with retrieval) | 2–5 s | Depends on LLM generation speed and chunk count |
| Query (gate: direct) | 0.5–1 s | Skips Pinecone; LLM-only response |
| Detection (background) | 0.6–1.2 s | 5 rules; Rules 4–5 require one embedding call each |
| Single repair cascade | 8–30 s | Depends on which strategy wins; S4 is slowest |
| Document ingestion | 5–60 s | Depends on document size and Ollama embedding throughput |
| Staleness check (50 samples) | 10–20 s | Requires 50 re-embedding calls |

> [!NOTE]
> These numbers will vary significantly based on your hardware, especially GPU availability for Ollama. CPU-only inference can be 3–5× slower for LLM calls.

---

## Troubleshooting

### Ollama not running or model not pulled

```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Fix:** Ensure Ollama is running (`ollama serve`) and you've pulled the required models:

```bash
ollama list                      # check what's installed
ollama pull mxbai-embed-large    # embedding model (required)
ollama pull mistral              # primary LLM (required)
```

### Pinecone index dimension mismatch

```
pinecone.exceptions.PineconeApiException: Vector dimension mismatch
```

**Fix:** Your Pinecone index must be created with **1024 dimensions** to match the mxbai-embed-large embedding model. Check your index configuration in the Pinecone console.

### SQLite lock errors

```
sqlalchemy.exc.OperationalError: database is locked
```

**Fix:** This can happen when the API server, auto-worker, and dashboard all write to the same SQLite file simultaneously. In most cases this is transient. If persistent:
- Ensure only one instance of each process is running
- For Docker, verify the `rag_db` volume is correctly shared across containers

### .env misconfiguration

```
pydantic_core._pydantic_core.ValidationError: ... field required
```

**Fix:** Copy `env_format.txt` to `.env` and ensure `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, and `PINECONE_NAMESPACE` are set. Don't leave them as placeholder values like `API_KEY` or `INDEX_NAME`.

### Docker container can't reach Ollama on host

```
ConnectionRefusedError — http://host.docker.internal:11434
```

**Fix:**
- **macOS / Windows:** `host.docker.internal` should work out of the box. Ensure Ollama is running on the host.
- **Linux:** Add the following to your `docker-compose.yml` under each service:
  ```yaml
  extra_hosts:
    - "host.docker.internal:host-gateway"
  ```
  Or use `--add-host=host.docker.internal:host-gateway` with `docker run`.

---

## License

This project was developed as part of an academic collaboration with HPE. License details will be added in a future release.

<!-- TODO: Add a LICENSE file (MIT, Apache-2.0, or as agreed with HPE) and update this section. -->

---

## Contributing

For questions, bug reports, or contributions, please open an issue on the [GitHub repository](https://github.com/Ayaan0777/Self-Oragnising-RAG).
