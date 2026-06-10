# Self-Organising RAG — Self-Healing Pipeline Documentation

> A RAG system that **monitors its own retrieval quality**, **detects degradation**, and **automatically repairs itself** by adaptively re-chunking and re-embedding failing documents — with full rollback safety.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Variable Chunking (Ingestion)](#2-variable-chunking-ingestion)
3. [Dynamic K — Adaptive Chunk Retrieval](#3-dynamic-k--adaptive-chunk-retrieval)
4. [Detection Metrics (MEASURE Stage)](#4-detection-metrics-measure-stage)
5. [Rate-Based Sliding Window (Maintenance Loop)](#5-rate-based-sliding-window-maintenance-loop)
6. [Adaptive Repair — DECIDE → ACT → PROBE → COMMIT/ROLLBACK](#6-adaptive-repair--decide--act--probe--commitrollback)
7. [Batch Repair](#7-batch-repair)
8. [Rollback Mechanism](#8-rollback-mechanism)
9. [Other Features](#9-other-features)
10. [File Reference](#10-file-reference)

---

## 1. Architecture Overview

The system operates as a **closed-loop feedback system** with 5 stages:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          QUERY TIME PIPELINE                           │
│                                                                        │
│  User Query → Embed → Retrieve 2×k chunks → Rerank → Dynamic K prune  │
│       → LLM generates answer → Log to DB → Run Detection Metrics      │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      BACKGROUND WORKER (every 30s)                     │
│                                                                        │
│  Sliding Window (last 50 queries) → Calculate flagged rate             │
│  If rate ≥ 30% → Trigger BATCH REPAIR on ALL unresolved events        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         REPAIR PIPELINE (per event)                    │
│                                                                        │
│  DECIDE (select chunk size) → BACKUP old chunks → DELETE old chunks    │
│  → RECHUNK with new params → INSERT new chunks → PROBE (re-query)     │
│  → If improved: COMMIT  |  If not: ROLLBACK (restore old chunks)      │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Design Principle**: The system never permanently loses data. Every repair attempt is backed by a full rollback mechanism — if rechunking doesn't improve the score, the original chunks are restored exactly as they were.

---

## 2. Variable Chunking (Ingestion)

**File**: `controllers/ingestion.py`

During document ingestion, text is split using `RecursiveCharacterTextSplitter` with variable-size chunks instead of fixed-size.

| Parameter       | Value | Rationale                                                    |
|-----------------|-------|--------------------------------------------------------------|
| `chunk_size`    | 1250  | Large enough to preserve context, fits in mxbai-embed-large's ~512 token window |
| `chunk_overlap` | 200   | Ensures no information is lost at chunk boundaries           |
| `min_size`      | 200   | Any chunk smaller than 200 chars is merged with its neighbour to avoid fragments |

### Min-Size Enforcement

After splitting, any chunk shorter than 200 characters is merged with its adjacent chunk. This prevents tiny, meaningless fragments from polluting the vector index.

```
Before merge:  ["This is a full chunk... (800 chars)", "end. (45 chars)", "Next section... (600 chars)"]
After merge:   ["This is a full chunk... end. (845 chars)", "Next section... (600 chars)"]
```

### Separators (Priority Order)

The splitter tries to break text at natural boundaries in this order:

```
"\n\n"  →  Paragraph breaks (highest priority)
"\n"    →  Line breaks
". "    →  Sentence endings
"? "    →  Question endings
"! "    →  Exclamation endings
"; "    →  Semicolons
", "    →  Commas
" "     →  Words (last resort)
```

---

## 3. Dynamic K — Adaptive Chunk Retrieval

**File**: `organiser/dynamic_k.py`

Instead of retrieving a fixed number of chunks (k) for every query, the system retrieves **2× the target k** from Pinecone, then prunes down based on **query complexity**.

### How It Works

```
Step 1: User sends query with target k (e.g., k=5)
Step 2: System retrieves 2×k = 10 chunks from Pinecone
Step 3: Analyse query complexity using heuristics
Step 4: Set effective k based on complexity:
        - Simple query → k stays at 5
        - Complex query → k expands up to 10 (2× target)
Step 5: Return top effective_k chunks to the LLM
```

### Query Complexity Heuristics

Complexity is determined by **code-based rules** (no LLM call, zero latency):

| Signal                    | Weight | Example                                         |
|---------------------------|--------|--------------------------------------------------|
| Multiple sub-questions    | +1     | "What is X and how does it relate to Y?"        |
| Comparison keywords       | +1     | "compare", "contrast", "difference between"    |
| Multi-part structure      | +1     | Contains "1.", "2." or "a)", "b)"               |
| Long query (>15 words)    | +1     | Detailed, multi-clause questions                 |
| Technical/domain terms    | +1     | "algorithm", "architecture", "implementation"   |

**Complexity Score → k Multiplier**:

```
Score 0-1  → Simple   → effective_k = target_k        (e.g., 5)
Score 2-3  → Medium   → effective_k = target_k × 1.5  (e.g., 8)
Score 4+   → Complex  → effective_k = target_k × 2    (e.g., 10)
```

### Why This Matters

- **Simple questions** ("What year was X founded?") need 3-5 focused chunks. Retrieving too many adds noise and confuses the LLM.
- **Complex questions** ("Compare the advantages and disadvantages discussed in chapters 3 and 5") need 8-10 chunks to gather enough evidence from multiple sections.

---

## 4. Detection Metrics (MEASURE Stage)

**File**: `detector/detectors.py`

After every query, 3 detection metrics + 1 bonus check are run to determine if the retrieval/answer quality is poor.

### Metric 1: Retrieval Precision

**Type**: Score-based (no LLM call)

Two checks:

1. **Proportion check**: At least 50% of retrieved chunks must have cosine similarity ≥ 0.45.
2. **Absolute floor**: The best chunk's score must be ≥ 0.55. This catches cases where all chunks have borderline scores (~0.46-0.49) — which happens when the index simply doesn't contain relevant content.

```
Example — Western Ghats query (no relevant docs ingested):
  Scores: [0.49, 0.47, 0.46, 0.46, 0.45, 0.45, 0.44, 0.44]
  Top-1 score: 0.49 < 0.55 → FLAGGED ✓
  
Example — University Library query (relevant docs exist):
  Scores: [0.68, 0.66, 0.64, 0.58, 0.56, 0.56, 0.55, 0.55]
  Top-1 score: 0.68 ≥ 0.55 → OK
  Proportion ≥ 0.45: 8/8 = 100% → OK
```

### Metric 2: Context Sufficiency

**Type**: LLM-based (1 LLM call)

The LLM is asked whether the retrieved context contains **relevant information** to answer the question. This only flags when the context is **completely irrelevant or missing critical information** — it does NOT flag broad questions that are "partially" answered.

```
Prompt: "Does the context contain RELEVANT information to answer this question?
         Reply 'NO' ONLY if the context is completely irrelevant or
         missing critical information needed to answer the question at all."
```

**Trigger**: LLM replies "NO" → `context_insufficient`

### Metric 3: Hallucination Rate

**Type**: LLM-based (1 LLM call)

The LLM checks if the generated answer **contradicts or contains factually wrong information** compared to the context. It does NOT flag extra details the LLM adds from its parametric knowledge — only outright fabrication or wrong facts.

```
Prompt: "Does the answer contain any information that CONTRADICTS or is
         FACTUALLY WRONG compared to the context above? Only flag if the
         answer says something the context explicitly disagrees with or
         fabricates specific facts (dates, numbers, names) that are wrong."
```

**Trigger**: LLM replies "YES" → `hallucination_detected`

### Bonus: LLM Self-Admitted Uncertainty

**Type**: Keyword-based (zero LLM calls)

If the LLM's own answer contains phrases like "does not mention", "no information", "context does not", etc., that's a clear signal of retrieval failure. This check adds `context_insufficient` if not already triggered.

```
Example: "The context provided does not mention the Western Ghats..."
         → Keyword match: "does not mention" → FLAGGED ✓
```

### Severity Assignment

| Detectors Triggered | Severity |
|---------------------|----------|
| 1 metric            | LOW      |
| 2 metrics           | MEDIUM   |
| 3 metrics           | HIGH     |

---

## 5. Rate-Based Sliding Window (Maintenance Loop)

**File**: `main.py`

The system does **NOT** repair after every bad query. Instead, it monitors the **overall failure rate** using a sliding window approach.

### How It Works

```
Every 30 seconds, the background worker:

1. Fetches the last 50 queries from the database
2. Counts how many were flagged
3. Calculates: failure_rate = flagged / total
4. If failure_rate ≥ 30% (i.e., ≥15 out of 50) → TRIGGER BATCH REPAIR
5. If failure_rate < 30% → Log "healthy" and sleep
```

### Configuration

| Parameter            | Value | Description                                    |
|----------------------|-------|------------------------------------------------|
| `WINDOW_SIZE`        | 50    | Number of recent queries to consider           |
| `FAILURE_RATE_THRESHOLD` | 0.30  | 30% flag rate triggers repair              |
| `CHECK_INTERVAL`     | 30s   | How often the worker checks                    |
| `MIN_QUERIES_BEFORE_CHECK` | 10 | Don't check until at least 10 queries exist |

### Why Rate-Based and Not Per-Query?

| Approach              | Problem                                                    |
|-----------------------|------------------------------------------------------------|
| Repair every bad query | Wastes resources — some queries are just bad queries, not index problems |
| Repair per-event      | Causes DB inconsistency — constant re-embedding fragments the index |
| **Rate-based (ours)** | Only repairs when there's a **systemic** problem — high failure rate indicates the index itself needs fixing, not just one query |

### Sliding Window Behaviour

The window uses **whatever queries exist** up to 50. It does NOT wait for exactly 50:

```
If 15 queries exist and 6 are flagged: 6/15 = 40% > 30% → REPAIR
If 50 queries exist and 10 are flagged: 10/50 = 20% < 30% → HEALTHY
If 50 queries exist and 20 are flagged: 20/50 = 40% > 30% → REPAIR
```

---

## 6. Adaptive Repair — DECIDE → ACT → PROBE → COMMIT/ROLLBACK

**Files**: `repair/orchestrator.py`, `repair/chunker.py`, `repair/reembedder.py`

When repair is triggered, each failing event goes through a 4-stage pipeline:

### Stage 1: DECIDE — Select Repair Parameters

The system analyses **which detector triggered** and the **query complexity** to determine the repair strategy:

| Condition                          | Chunk Size | Overlap | Reason              |
|------------------------------------|-----------|---------|----------------------|
| `context_insufficient` triggered    | 1500      | 300     | Need MORE context per chunk |
| `hallucination_detected` triggered  | 400       | 80      | Need LESS noise, more focused chunks |
| Complex query + low precision       | 1500      | 300     | Complex questions need bigger chunks |
| Simple query + low precision        | 400       | 80      | Simple questions need focused chunks |
| Default / mixed signals             | 800       | 150     | Balanced middle ground |

**Key Insight**: The chunk size used during repair is **different from ingestion** (1250). If the context was insufficient, we make chunks bigger (1500) to capture more surrounding context. If the LLM was hallucinating, we make chunks smaller (400) to reduce noise and keep content focused.

### Stage 2: ACT — Delete Old → Rechunk → Insert New

```
1. Fetch the old chunk IDs associated with the failing query
2. Concatenate all old chunk text into a single block
3. DELETE old chunks from Pinecone
4. RECHUNK the text using the DECIDED parameters (different from ingestion)
5. INSERT new chunks into Pinecone
```

### Stage 3: PROBE — Verify Improvement

After inserting new chunks, the system re-runs the **exact same query** against the updated index and compares the new top-1 score against the original:

```
score_before = 0.4913  (original failing query)
score_after  = 0.6721  (after rechunking)
improved = score_after > score_before  → YES → COMMIT
```

### Stage 4: COMMIT or ROLLBACK

- **If improved**: Mark the event as resolved, log the repair report with `resolved=True`
- **If NOT improved**: **ROLLBACK** — remove the new chunks, restore the exact original chunks from backup, log with `rolled_back=True`

---

## 7. Batch Repair

**File**: `main.py`

When the sliding window triggers repair, it doesn't just fix one event — it repairs **ALL unresolved events** in a single batch.

### Batch Repair Flow

```
Rate exceeded (e.g., 28/48 = 58%) → Trigger batch repair

🔧 [Batch Repair] Starting repair of 15 events...
   [1/15] Repairing event #3...
   [1/15] ✅ Event #3 COMMITTED (increase_context, chunk_size=1500, 0.491→0.672)
   [2/15] Repairing event #7...
   [2/15] 🔄 Event #7 ROLLED BACK (reduce_noise, chunk_size=400, 0.521→0.498)
   [3/15] Repairing event #12...
   [3/15] ✅ Event #12 COMMITTED (increase_context, chunk_size=1500, 0.445→0.623)
   ...
📊 [Batch Repair] Done — Committed: 9, Rolled back: 4, Errors: 2
```

### What Happens to Each Event

- **Each event** gets its own DECIDE stage (chunk size may differ per event)
- **Each event** gets its own PROBE (score is checked individually)
- **Each event** gets its own COMMIT/ROLLBACK decision

This means in a single batch, some events may be committed (improved) and others rolled back (no improvement) — each is independent.

### After Batch Repair

The failure rate naturally drops because:
- Successfully repaired events are marked `resolved=True` → no longer count as flagged
- Rolled-back events keep their `resolved=False` but are marked `unfixable=True` → won't be retried

---

## 8. Rollback Mechanism

**File**: `repair/reembedder.py`

The rollback mechanism ensures the vector index is **never permanently corrupted** by a bad repair.

### How Rollback Works

```
Step 1: BACKUP — Fetch all old vectors from Pinecone (IDs + embeddings + metadata)
                 Store in memory dictionary: {id: {values, metadata}}

Step 2: DELETE — Remove old vectors from Pinecone by ID

Step 3: INSERT — Add new rechunked vectors to Pinecone, get new IDs

Step 4: PROBE  — Re-query Pinecone with the original query
                 Compare new score vs old score

Step 5a: COMMIT (if improved)
         - New chunks stay in Pinecone ✓
         - Backup is discarded
         - Event marked as resolved

Step 5b: ROLLBACK (if NOT improved)
         - DELETE the new chunks (by new IDs)
         - RE-INSERT the old chunks from backup (exact original vectors)
         - Index is restored to its exact pre-repair state
         - Event marked as rolled_back
```

### Why This Approach?

- **In-memory backup** is the only safe way to guarantee rollback integrity in Pinecone (Pinecone has no built-in transaction/undo)
- **Deleting before inserting** ensures the PROBE tests against a clean index without interference from old chunks
- The backup contains the **exact original embedding vectors**, not re-computed ones — so rollback produces a byte-identical restoration

---

## 9. Other Features

### LLM-Based Reranking
- **File**: `services/reranker.py`
- After retrieval, the LLM re-scores and reorders chunks by relevance
- One batch LLM call (not per-chunk), safe fallback on failure
- Enabled by default (`rerank=True`)

### Add Chunks (Dashboard)
- **Dashboard page**: Paste raw text → preview recursive chunks → optionally ingest to Pinecone
- Uses the same chunking params as ingestion (1250/200/min 200)

### Repair History (Dashboard)
- Full table of all repair attempts with: strategy, chunk size, score before/after, committed/rolled back status, duration

### Re-Query Comparison (Dashboard)
- For resolved events, a "RE-QUERY NOW" button runs the original query again
- Shows side-by-side: original answer (before repair) vs current answer (after repair)

### Repair Reports (API)
- `GET /api/v1/repair-reports` — returns full repair history with all fields
- `POST /api/v1/repair/{event_id}` — manually trigger repair on a specific event (strategy is auto-selected by DECIDE stage)

---

## 10. File Reference

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app + background maintenance worker (sliding window + batch repair) |
| `controllers/ingestion.py` | Variable chunking (1250/200, min 200 enforcement) |
| `controllers/retrieval.py` | Query pipeline: embed → retrieve 2×k → rerank → dynamic k → LLM |
| `organiser/dynamic_k.py` | Query complexity analysis + adaptive k sizing |
| `detector/detectors.py` | 3 detection metrics + LLM uncertainty check |
| `repair/orchestrator.py` | DECIDE → ACT → PROBE → COMMIT/ROLLBACK pipeline |
| `repair/chunker.py` | Adaptive rechunking with variable params per event |
| `repair/reembedder.py` | Backup, delete, insert, rollback, probe operations |
| `api/routes.py` | REST endpoints for ingest, query, repair, repair-reports |
| `dashboard/app.py` | Streamlit dashboard with all monitoring pages |
| `db/models.py` | SQLAlchemy models (QueryLog, LowRecallEvent, RepairReport) |
| `db/session.py` | DB init + schema migration for new columns |
| `auto_chunker/pipeline.py` | Recursive chunking for the Add Chunks feature |
| `services/reranker.py` | LLM-based chunk reranking |
