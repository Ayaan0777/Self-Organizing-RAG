<<<<<<< HEAD
# Auto-RAG — Self-Healing Retrieval-Augmented Generation

A local-first RAG system with **automatic failure detection**, **self-healing repair**, and a **Streamlit diagnostic dashboard**. Built on FastAPI, PGVector, Ollama, and SQLAlchemy.

---

## Features

| Module                  | What it does                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| **Ingestion**           | Upload PDF / DOCX / TXT files → cleaned, chunked, and embedded into PGVector                |
| **Retrieval**           | Query the vector store, get an LLM-generated answer with top-K similarity scores            |
| **Evaluation**          | Run batch evaluation (ROUGE-L, Semantic Similarity, Context↔Query/GT similarity)            |
| **Query Logger**        | Every query is persisted with scores, latency, and retrieved chunks                         |
| **Failure Detector**    | Rule-based detection flags low-recall queries (low score, score drop, LLM uncertainty)      |
| **Self-Healing Repair** | Re-chunks and re-embeds affected documents using semantic / LLM / entropy strategies        |
| **Dashboard**           | Streamlit UI — Overview, Query Diagnostics, Low-Recall Events, Repair History, Eval History |

---

## Prerequisites

- **Python 3.10+**
- **PostgreSQL** with the [`pgvector`](https://github.com/pgvector/pgvector) extension installed
- **Ollama** running locally ([ollama.com](https://ollama.com))

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull Ollama models

```bash
ollama pull mxbai-embed-large
ollama pull mistral
```

### 4. Configure environment variables

Create a `.env` file in the project root (or edit the existing one):

```env
# Vector DB (PGVector)
DATABASE_URL=postgresql+psycopg://postgres:<password>@localhost:5432/rag
COLLECTION_NAME=rag_collection

# Local Embeddings (Ollama)
EMBEDDING_MODEL_NAME=mxbai-embed-large
OLLAMA_BASE_URL=http://localhost:11434

# LLM
LLM_PROVIDER=ollama
LLM_MODEL_NAME=mistral
```

### 5. Ensure PostgreSQL + pgvector are ready

Make sure your Postgres database exists and the `pgvector` extension is enabled:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## Running

### Start the API server

```bash
uvicorn main:app --reload
```

The database tables are created automatically on startup via `init_db()`.

### Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## API Endpoints

All endpoints are prefixed with `/api/v1`.

| Method | Endpoint                               | Description                                                            |
| ------ | -------------------------------------- | ---------------------------------------------------------------------- |
| `POST` | `/ingest`                              | Upload a file (PDF, DOCX, TXT) for ingestion                           |
| `POST` | `/query`                               | Query the RAG pipeline — returns answer, contexts, scores, and log ID  |
| `POST` | `/evaluate`                            | Single-question evaluation (question + ground truth)                   |
| `GET`  | `/logs?limit=50`                       | Recent query logs with scores and flag status                          |
| `GET`  | `/events`                              | All low-recall events                                                  |
| `POST` | `/repair/{event_id}?strategy=semantic` | Trigger a repair for a specific event (`semantic` / `llm` / `entropy`) |

### Example — Ingest a document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@dataset/contexts.docx"
```

### Example — Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

---

## Batch Evaluation

Run the evaluation script against a dataset of question–answer pairs:

```bash
python run_evaluation.py                    # default namespace
python run_evaluation.py my_namespace       # custom namespace
```

**Metrics computed:**

- **ROUGE-L** — lexical overlap with ground truth
- **Semantic Similarity** — cosine similarity between answer and ground truth embeddings
- **Context↔Query Similarity** — how relevant the retrieved chunks are to the question
- **Context↔Ground Truth Similarity** — how relevant the retrieved chunks are to the expected answer

Results are saved to `results/` as CSV and JSON files, and a snapshot is persisted to the database for dashboard tracking.

---

## Detection Rules

The detector module flags queries automatically based on these rules:

| Rule              | Triggers When                                                            | Threshold     |
| ----------------- | ------------------------------------------------------------------------ | ------------- |
| `low_top_score`   | Best chunk similarity is below threshold                                 | **0.45**      |
| `score_drop`      | Gap between rank-1 and rank-K scores is too large                        | **0.3**       |
| `llm_uncertainty` | LLM response contains hedging language ("I don't know", "unclear", etc.) | keyword match |

When one or more rules trigger, a `LowRecallEvent` is created with severity (`LOW`, `MEDIUM`, or `HIGH`) based on how many rules fired.

---

## Self-Healing Repair

Triggered via the `/repair/{event_id}` endpoint or the dashboard. The repair pipeline:

1. Loads the original failing query from the event
2. Retrieves the affected source document
3. Re-chunks using the selected strategy:
   - **`semantic`** — semantic-aware chunking
   - **`llm`** — LLM-guided chunking
   - **`entropy`** — entropy-based chunking
4. Deletes old vectors and inserts new embeddings
5. Re-runs the original query to measure score improvement
6. Marks the event as resolved if the score improves by ≥ 5%

---

## Project Structure

```
test/
├── .env                        ← DB + model config
├── .gitignore
├── main.py                     ← FastAPI entry point
├── run_evaluation.py           ← batch evaluation script
├── dataset_ingest.py           ← standalone ingestion helper
├── requirements.txt
├── README.md
│
├── api/
│   └── routes.py               ← 6 API routes
│
├── controllers/
│   ├── evaluation.py
│   ├── ingestion.py            ← file upload → clean → chunk → embed
│   └── retrieval.py            ← query → retrieve → answer → log → detect
│
├── services/
│   └── llm_factory.py          ← LLM + embeddings + vector store factory
│
├── db/                         ← SQLAlchemy models + session
│   ├── models.py               ← QueryLog, LowRecallEvent, RepairReport, EvalSnapshot
│   └── session.py              ← engine + init_db() + get_session()
│
├── logger/
│   └── query_logger.py         ← log_query() + update_log_eval_metrics()
│
├── detector/
│   └── detectors.py            ← 3 detection rules + run_detectors()
│
├── repair/                     ← self-healing pipeline
│   ├── chunker.py              ← semantic / LLM / entropy re-chunking
│   ├── reembedder.py           ← delete old vectors + insert new
│   └── orchestrator.py         ← full repair loop
│
├── dashboard/
│   └── app.py                  ← 5-tab Streamlit diagnostic dashboard
│
├── dataset/                    ← evaluation datasets (CoQa, MS MARCO, etc.)
└── results/                    ← historical evaluation results (CSV/JSON)
```

---

## Tech Stack

- **FastAPI** — REST API framework
- **Ollama** — local LLM and embedding inference
- **PGVector** — PostgreSQL vector similarity search
- **LangChain** — document loading, splitting, retrieval chains
- **SQLAlchemy** — query logging and event persistence
- **Streamlit** — diagnostic dashboard UI
- **RAGAS / NumPy** — evaluation metrics
=======
📌 Self-Organising / Self-Healing RAG

Phase-wise Project Plan & Tech Stack

1️⃣ What is RAG (Short & Accurate)

Retrieval-Augmented Generation (RAG) is a system where an LLM generates answers using externally retrieved documents instead of relying only on its internal knowledge.

Core idea:

Retrieve relevant context → inject into prompt → generate grounded answer

2️⃣ Why RAG
Problem in LLMs	RAG Solution
Hallucinations	Grounding with real documents
Outdated knowledge	Dynamic retrieval
Private data access	Use internal documents
Costly retraining	No model retraining
3️⃣ Basic RAG Workflow (Minimal)
User Query
 → Query Embedding
 → Vector Search (Top-K)
 → Retrieved Chunks
 → Prompt Augmentation
 → LLM Answer
4️⃣ Core Components (What & Why)
Component	Why Needed	Tech Options
LLM	Generate final answer	GPT / Llama / Mistral
Embedding Model	Semantic search	OpenAI / BGE / E5
Vector DB	Fast similarity search	FAISS / Chroma
Chunking	Better retrieval accuracy	Token-based splitting
RAG Framework	Orchestration	LangChain / LlamaIndex
🚀 Phase-Wise Development Plan
🔹 Phase 1: Basic RAG (Foundation)
Goal

Build a working RAG pipeline.

Tasks

Prepare documents

Chunk text

Generate embeddings

Store in vector DB

Retrieve Top-K chunks

Generate answer using LLM

Tech Stack & Why
Tech	Why
Python	Ecosystem + ML support
LangChain	Fast RAG prototyping
FAISS / Chroma	Lightweight local vector DB
OpenAI / Llama	High-quality generation
Sentence embeddings	Semantic similarity

✅ Outcome: Working RAG system

🔹 Phase 2: Improved Retrieval (Quality Boost)
Goal

Increase retrieval relevance.

Tasks

Improve chunking strategy

Use better embeddings

Add metadata filtering

Implement Top-K tuning

Tech Stack & Why
Tech	Why
BGE / E5 embeddings	Better retrieval quality
Metadata filters	Context narrowing
Reranking models	Improve Top-K relevance

✅ Outcome: Fewer wrong contexts

🔹 Phase 3: Self-Healing RAG (Correction Layer)
Goal

Automatically detect and fix bad answers.

Tasks

Add self-evaluation step

Detect low-confidence answers

Retry retrieval

Regenerate answer

Architecture
Answer → Self-Check → Retry? → Refine → Final Output
Tech Stack & Why
Tech	Why
LLM self-critique	Detect hallucination
Retry logic	Automatic correction
Prompt refinement	Better answers

✅ Outcome: Reduced hallucination

🔹 Phase 4: Self-Organising RAG (Adaptive Intelligence)
Goal

System decides how to retrieve and when to retry.

Tasks

Decide when retrieval is needed

Dynamically re-query

Adapt retrieval strategy

Track failure patterns

Tech Stack & Why
Tech	Why
SELF-RAG concepts	Retrieval decision logic
Reflection tokens	Control flow
LangGraph	Agent-style execution
Feedback loops	Continuous improvement

✅ Outcome: Adaptive & intelligent RAG

🔹 Phase 5: Evaluation & Scaling (Production)
Goal

Make system reliable and scalable.

Tasks

Measure retrieval accuracy

Measure answer faithfulness

Scale vector DB

Add caching & monitoring

Tech Stack & Why
Tech	Why
Recall@K / MRR	Retrieval evaluation
Faithfulness metrics	Answer grounding
Milvus / Qdrant	Scalable vector DB
FastAPI	Production API
Redis	Cache retrieval

✅ Outcome: Production-ready system

📦 Final Tech Stack Summary
Layer	Tech
Language Model	GPT / Llama
Embeddings	BGE / E5
Vector DB	FAISS → Milvus
RAG Framework	LangChain / LlamaIndex
Backend	FastAPI
Evaluation	LangSmith / Custom metrics
>>>>>>> d26aaf62b01a6abee71f08b2a67a25270fdcbf16
