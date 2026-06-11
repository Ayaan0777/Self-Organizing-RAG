# Self-Healing RAG (SH2) — Feature Comparison Checklist

> **Purpose:** A structured feature list for comparing this codebase against other Self-Healing / Self-Organising RAG implementations.

---

## 1. Architecture & Tech Stack

| Feature | Details |
|---|---|
| Framework | FastAPI (Python) |
| LLM Provider | Ollama (local) — default model: `mistral` |
| Embedding Model | `mxbai-embed-large` (1024 dims) via Ollama |
| Vector Store | Pinecone (cloud) with namespace isolation |
| Database | SQLite via SQLAlchemy ORM |
| Chunking Library | LangChain `RecursiveCharacterTextSplitter` |
| Document Loaders | PDF (`PyPDFLoader`), DOCX (`Docx2txtLoader`), TXT (`TextLoader`) |
| Dashboard | Streamlit (`dashboard/app.py`) |
| API Prefix | `/api/v1/` |

---

## 2. Auto-Repair (Self-Healing) Pipeline

### 2.1 Overall Repair Architecture

- [x] **4-Stage pipeline:** `DECIDE → ACT → PROBE → COMMIT/ROLLBACK`
- [x] **Autonomous background worker** — runs on a sliding window schedule
- [x] **Rate-based triggering** — not per-query, but batch-based on failure rate
- [x] **Safe rollback** — old vectors are NEVER permanently lost
- [x] **Repair reports persisted** to SQLite (`RepairReport` table)

### 2.2 Rate-Based Trigger Mechanism

| Parameter | Value |
|---|---|
| Check interval | Every **30 seconds** |
| Sliding window | Last **50 queries** |
| Failure rate threshold | **30%** (15/50 flagged triggers repair) |
| Minimum queries before check | **10** |
| Evaluation mode | Background healing DISABLED when `ENV=evaluation` |

### 2.3 DECIDE Stage — Repair Strategy Selection

- [x] **Detector-based decisions** (highest priority):
  - `context_insufficient` → `chunk_size=1500`, `overlap=300` (increase context)
  - `hallucination_detected` → `chunk_size=400`, `overlap=80` (reduce noise)
- [x] **Query-complexity-based decisions** (fallback):
  - Complex query → `chunk_size=1500`, `overlap=300`
  - Simple query → `chunk_size=400`, `overlap=80`
  - Medium query → `chunk_size=800`, `overlap=150` (default)
- [x] **Always different from ingestion** — repair NEVER uses ingestion params (`1250/200`)
- [x] **Query complexity classification** via regex heuristics:
  - Complex patterns: `compare`, `difference`, `vs`, `explain`, `describe`, multiple `?`
  - Simple patterns: `who`, `when`, `where`, `how much/many`
  - Word count ≥ 15 → complex

### 2.4 BACKUP Stage

- [x] Old vectors fetched from Pinecone **into memory** before any deletion
- [x] Includes full vector data: `id`, `values` (embeddings), `metadata`
- [x] Batched fetch (1000 IDs per batch, Pinecone limit)

### 2.5 ACT Stage — Delete → Rechunk → Insert

- [x] Old chunks deleted from Pinecone by vector ID
- [x] Text re-chunked with DECIDE-selected params
- [x] New chunks embedded and inserted into Pinecone
- [x] Rechunking verification logging (old count → new count, new sizes)

### 2.6 PROBE Stage — Score Verification

- [x] Re-runs the original failing query against Pinecone post-repair
- [x] Returns top-1 similarity score (0.0–1.0)
- [x] Uses `similarity_search_with_score(query, k=5)`
- [x] Takes the **max** of top-5 scores as the probe score

### 2.7 COMMIT / ROLLBACK

- [x] **Improvement threshold:** Score must beat original by ≥ **0.05**
- [x] **COMMIT:** New chunks stay, backup discarded, event marked `resolved=True`
- [x] **ROLLBACK:** New chunks deleted, old chunks restored from backup via `upsert`, event marked `unfixable=True`
- [x] Rollback processes in batches (100 vectors per upsert)
- [x] Rollback success/failure logged

### 2.8 Repair Report Logging

Each repair attempt records:
- [x] `event_id` — which failure event was repaired
- [x] `strategy_used` — e.g., `increase_context`, `reduce_noise`, `complex_query`
- [x] `chunk_size_used` — the chunk_size parameter used
- [x] `repair_reason` — human-readable reason
- [x] `chunks_before` / `chunks_after` — count comparison
- [x] `score_before` / `score_after` — quality comparison
- [x] `resolved` / `rolled_back` — outcome
- [x] `duration_ms` — repair duration
- [x] `timestamp` — when repair happened

---

## 3. Retrieval Pipeline

### 3.1 Full Retrieval Flow

```
Query → Retrieval Gate → Retrieval → Reranking → Dynamic K → Query Reformulation → Answer Generation → Logging → Detection
```

### 3.2 Retrieval Gating (Pre-Retrieval)

- [x] **Two-tier classification:**
  1. Fast pattern matching for obvious greetings/chitchat (e.g., `hi`, `hello`, `thanks`)
  2. LLM classification for ambiguous queries (`RETRIEVE` vs `DIRECT`)
- [x] Short non-question queries (≤ 2 words, no question words) → direct answer
- [x] Fallback: defaults to RETRIEVE on any error
- [x] Direct answers bypass Pinecone entirely — answered by LLM alone

### 3.3 Vector Retrieval

- [x] **Over-fetching:** Fetches `min(k*2, 15)` chunks for Dynamic K to prune
- [x] Pinecone `similarity_search_with_score` — returns cosine similarity (0–1)
- [x] Optional **metadata filter** support
- [x] Namespace isolation

### 3.4 LLM-Based Reranking

- [x] Single batch LLM call (not per-chunk)
- [x] LLM receives all chunks with previews (first 300 chars) and returns a JSON ranking
- [x] Robust parsing: handles JSON in markdown code blocks, partial rankings, duplicates
- [x] Missing indices appended in original order
- [x] Can be disabled via `rerank=False` parameter
- [x] Falls back to original order on any failure

### 3.5 Dynamic K Selection

Three-stage algorithm:
- [x] **Stage 1 — Query Complexity Analysis:**
  - Comparison/multi-part → K bounds `[4, 10]`
  - Broad/exploratory → K bounds `[3, 8]`
  - Specific factual → K bounds `[2, 5]`
  - Long queries (≥ 15 words) → K bounds `[3, 8]`
  - Default → K bounds `[3, 6]`
- [x] **Stage 2 — Absolute Score Pruning:**
  - Removes chunks with score < **0.25** (noise floor)
- [x] **Stage 3 — Score Cliff Detection:**
  - Finds largest drop between consecutive scores > **0.12** threshold
  - Cuts AFTER the cliff point
- [x] **Minimum K = 2** (always returns at least 2 chunks)
- [x] Falls back to `target_k` on failure

### 3.6 Query Reformulation

- [x] Triggers when top-1 score < **0.45**
- [x] LLM rephrases the query for better vector search matching
- [x] Max **1 reformulation attempt** (no infinite loops)
- [x] Only replaces results if rephrased query scores **genuinely better**
- [x] Falls back to original results on failure

### 3.7 Answer Generation

- [x] LangChain `create_stuff_documents_chain` — single-pass context stuffing
- [x] Prompt: `"Answer the question based only on the context provided"`
- [x] Uses Ollama Mistral with `temperature=0.2`

---

## 4. Chunking Strategies

### 4.1 Ingestion Chunking

| Parameter | Value |
|---|---|
| Method | `RecursiveCharacterTextSplitter` |
| Chunk Size | **1250 chars** |
| Chunk Overlap | **200 chars** (~16% of max) |
| Min Chunk Size | **200 chars** (enforced by merging) |
| Separators | `\n\n`, `\n`, `. `, `? `, `! `, `; `, `, `, ` `, `""` |
| Text Cleaning | Regex: collapse `\s+` → single space, strip |

### 4.2 Repair Chunking (Adaptive)

| Strategy | Chunk Size | Overlap | Trigger |
|---|---|---|---|
| `increase_context` | **1500** | 300 | `context_insufficient` detector |
| `reduce_noise` | **400** | 80 | `hallucination_detected` detector |
| `complex_query` | **1500** | 300 | Query complexity = complex |
| `simple_query` | **400** | 80 | Query complexity = simple |
| `default_repair` | **800** | 150 | Fallback |

- [x] **Repair ALWAYS uses different params from ingestion** (never 1250/200)
- [x] Same `RecursiveCharacterTextSplitter` with same separator hierarchy
- [x] Min chunk floor: **100 chars** for small chunks (≤400), **150 chars** otherwise
- [x] All chunk sizes within `mxbai-embed-large`'s 512-token window (1800 chars ≈ 450 tokens max)
- [x] Metadata tagged with `strategy: "adaptive_repair"`, `repair_chunk_size`, `repair_reason`

### 4.3 Auto-Chunker Module (Standalone)

- [x] Simplified single-strategy chunker using `RecursiveCharacterTextSplitter`
- [x] Same params as ingestion (1250/200/200)
- [x] Available via `/api/v1/auto-chunk` endpoint
- [x] Optional Pinecone ingestion (`ingest: true`)

### 4.4 Advanced Chunking Strategies (Available but not in main pipeline)

- [x] **Semantic Segmentation Chunker:**
  - Embeds each sentence, finds topic boundaries where consecutive similarity drops below threshold (default 0.65)
  - Groups sentences between boundaries into chunks
- [x] **Cluster Chunker:**
  - Agglomerative clustering on sentence embeddings
  - Cosine distance matrix with distance threshold (default 0.5)
  - Preserves original document order within clusters
- [x] **Adaptive Chunk Sizer:**
  - Post-processor: merges small chunks (< 200 tokens), splits large chunks (> 2000 tokens)
  - Token estimation: `len(text) / 4`

---

## 5. Failure Detection

### 5.1 Detection Metrics (3 Metrics)

| # | Metric | Method | Threshold | Detector Tag |
|---|---|---|---|---|
| 1 | **Retrieval Precision** | Score-based: (a) best chunk < 0.55 OR (b) < 50% of chunks ≥ 0.45 | 0.55 / 0.45 / 50% | `low_retrieval_precision` |
| 2 | **Context Sufficiency** | LLM judges if context can answer query (lenient: "at least some relevant facts") | Binary YES/NO | `context_insufficient` |
| 3 | **Hallucination Rate** | LLM checks if answer contradicts context (lenient: ignores supplementary info) | Binary YES/NO | `hallucination_detected` |
| Bonus | **LLM Uncertainty** | Keyword scan in answer text (14 phrases like "does not mention", "no information") | Phrase match | → `context_insufficient` |

### 5.2 Detection Thresholds

| Parameter | Value |
|---|---|
| `RELEVANCE_SCORE_FLOOR` | 0.45 (chunks below this are irrelevant) |
| `MIN_PRECISION_RATIO` | 0.50 (at least 50% of top-k must be relevant) |
| `TOP_SCORE_FLOOR` | 0.55 (best chunk score absolute minimum) |

### 5.3 Severity Classification

| Detectors Fired | Severity |
|---|---|
| 1 | LOW |
| 2 | MEDIUM |
| 3+ | HIGH |

### 5.4 Detection Behavior

- [x] Runs automatically after every query (`run_detectors(log_id)`)
- [x] Never blocks the response (fire-and-forget)
- [x] Creates `LowRecallEvent` with triggered detector list
- [x] Marks `QueryLog.flagged = True`
- [x] Skipped for direct-answer queries (no retrieval = no detection)
- [x] Fail-safe: LLM errors in detectors → don't flag (false negatives preferred over false positives)

---

## 6. Evaluation Metrics

### 6.1 Answer Quality Metrics

| Metric | Method |
|---|---|
| **ROUGE-L** | LCS-based F1 between prediction and ground truth (normalized, lowercased, punctuation-stripped) |
| **Semantic Similarity** | Cosine similarity between answer embedding and ground truth embedding |

### 6.2 Context Quality Metrics

| Metric | Method |
|---|---|
| **Context–Question Similarity** | Mean cosine sim between question embedding and each context chunk embedding |
| **Context–Ground Truth Similarity** | Mean cosine sim between ground truth embedding(s) and each context chunk embedding |
| **Best Context–Question Similarity** | Max of per-chunk question similarities |
| **Best Context–GT Similarity** | Max of per-chunk ground truth similarities |

### 6.3 Per-Query Logged Metrics

| Field | Description |
|---|---|
| `top_k_scores` | JSON list of cosine similarity scores for each retrieved chunk |
| `ctx_q_sim` | Average of top-k scores (context↔question similarity) |
| `answer_sem_sim` | Semantic similarity between answer and ground truth (set by evaluation) |
| `latency_ms` | End-to-end query latency in milliseconds |
| `flagged` | Boolean — whether any detector triggered |

### 6.4 Evaluation Snapshots (Persisted)

| Field | Description |
|---|---|
| `namespace` | Pinecone namespace evaluated |
| `llm` / `embeddings` | Model names used |
| `rouge_l` | Average ROUGE-L across all questions |
| `sem_sim` | Average semantic similarity |
| `ctx_q_sim` | Average context–question similarity |
| `ctx_gt_sim` | Average context–ground truth similarity |
| Results CSV | Saved to `results/evaluation_results_{namespace}.csv` |

---

## 7. Auto-Indexer (Index Maintenance)

### 7.1 Components

| # | Component | Description |
|---|---|---|
| 1 | **Staleness Detector** | Samples random vectors, re-embeds text with current model, compares cosine similarity. Drift threshold: **0.95** |
| 2 | **Partial Re-embedder** | Re-embeds only stale vectors via Pinecone `upsert` (in-place update, no delete) |
| 3 | **Index Refresher** | Upserts new/changed chunks, detects and deletes orphaned vectors (empty text < 10 chars) |
| 4 | **Consistency Checker** | Reports total vectors, dimension, metadata integrity %, empty text count, average retrieval score |

### 7.2 Full Refresh Pipeline

```
Consistency Check → Staleness Detection → Auto Re-embedding → Final Consistency Check
```

### 7.3 Health Status

| Status | Condition |
|---|---|
| `HEALTHY` | Metadata integrity > 90% AND vector count > 0 |
| `DEGRADED` | Metadata integrity ≤ 90% |
| `EMPTY` | Vector count = 0 |

---

## 8. Feedback & Adaptive Strategy

### 8.1 Feedback Loop Analyser

- [x] **Read-only** — analyses data and suggests changes, never auto-modifies thresholds
- [x] Reports: detector fire rates, strategy success rates, threshold suggestions, overall health score
- [x] Health score formula: `flag_score (0-50) + resolution_score (0-50)`
  - `flag_score = max(0, 50 - flag_rate*100)`
  - `resolution_score = resolution_rate * 50`
- [x] Threshold suggestions:
  - Flag rate > 50% → suggest raising thresholds
  - Flag rate < 5% → suggest lowering thresholds
  - Detector fires > 30% → flag for review
  - Resolution rate < 30% → suggest re-ingestion

### 8.2 Adaptive Strategy Selector

- [x] Analyses historical `RepairReport` outcomes
- [x] Ranks strategies by: success_rate (primary), avg_improvement (secondary)
- [x] Default order: `["semantic", "entropy", "llm"]`
- [x] New/unknown strategies appended at end
- [x] Falls back to default on no history or errors

---

## 9. Data Model (SQLite Tables)

| Table | Purpose |
|---|---|
| `autorag_query_log` | Every user query with scores, chunks, response, latency, flag status |
| `autorag_low_recall_events` | Detected failures with triggered detectors, severity, resolution status |
| `autorag_repair_reports` | Every repair attempt with strategy, score change, duration, outcome |
| `autorag_eval_snapshots` | Evaluation run summaries with averaged metrics |

---

## 10. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/ingest` | Upload & ingest document (PDF/DOCX/TXT) |
| POST | `/api/v1/query` | RAG query with reranking, dynamic K, reformulation |
| POST | `/api/v1/auto-chunk` | Chunk raw text, optionally ingest |
| POST | `/api/v1/evaluate-local` | Run evaluation with uploaded Q&A dataset |
| POST | `/api/v1/repair/{event_id}` | Manually trigger repair for a specific event |
| GET | `/api/v1/logs` | Recent query logs |
| GET | `/api/v1/events` | Low recall events (unresolved by default) |
| GET | `/api/v1/repair-reports` | Repair history |
| GET | `/api/v1/eval-history` | Evaluation snapshots |
| GET | `/api/v1/index/health` | Index consistency check |
| GET | `/api/v1/index/staleness` | Stale embedding detection |
| POST | `/api/v1/index/refresh` | Full auto-indexer refresh |
| GET | `/api/v1/feedback/analysis` | System health analysis & threshold suggestions |

---

## 11. Safety & Fault Tolerance

- [x] **Rollback guarantee:** Old vectors backed up before deletion, restored on failed repair
- [x] **Fail-safe detectors:** LLM errors in detection → don't flag (avoid false positives)
- [x] **Graceful fallbacks:** Reranker, Dynamic K, Query Reformulator, Retrieval Gate all fall back to original results on failure
- [x] **No infinite loops:** Max 1 reformulation attempt
- [x] **Batch processing:** All Pinecone operations batched (50–1000 per batch)
- [x] **Non-blocking detection:** Detectors run after response is returned
- [x] **Evaluation mode:** Background self-healing can be disabled via `ENV=evaluation`
- [x] **Unfixable marking:** Failed repairs mark events as `unfixable=True` to prevent infinite retry

---

## 12. Quick Comparison Checklist

Use this checklist to compare against other RAG implementations:

| Feature | SH2 | Other |
|---|---|---|
| Self-healing repair pipeline | ✅ | |
| Automatic failure detection (3 metrics) | ✅ | |
| Rollback on failed repair | ✅ | |
| Rate-based batch repair triggering | ✅ | |
| Adaptive chunk sizing during repair | ✅ | |
| Query complexity → chunk size mapping | ✅ | |
| Detector-driven repair strategy | ✅ | |
| LLM-based reranking | ✅ | |
| Dynamic K selection (3-stage) | ✅ | |
| Query reformulation on poor retrieval | ✅ | |
| Retrieval gating (skip retrieval for chitchat) | ✅ | |
| Embedding staleness detection | ✅ | |
| Partial re-embedding (upsert, no wipe) | ✅ | |
| Index consistency checking | ✅ | |
| Orphaned vector cleanup | ✅ | |
| Feedback loop with threshold suggestions | ✅ | |
| Adaptive strategy ordering from history | ✅ | |
| ROUGE-L evaluation | ✅ | |
| Semantic similarity evaluation | ✅ | |
| Context relevance metrics | ✅ | |
| Per-query logging with full metadata | ✅ | |
| Repair report with before/after scores | ✅ | |
| Evaluation history persistence | ✅ | |
| Health score (0–100) | ✅ | |
| Semantic segmentation chunking | ✅ (module) | |
| Agglomerative clustering chunking | ✅ (module) | |
| Minimum chunk size enforcement | ✅ | |
| Multi-format document support (PDF/DOCX/TXT) | ✅ | |
| Namespace-based data isolation | ✅ | |
| Dashboard (Streamlit) | ✅ | |
