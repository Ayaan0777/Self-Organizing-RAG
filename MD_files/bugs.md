# Self-Organising RAG — Bug Audit

Comprehensive findings from the end-to-end code review. Ordered by severity
within each tier. Each entry includes a file:line anchor, what's broken, why
it matters, and a proposed fix.

Tags:
- **NEW** — surfaced or introduced during the cascade refactor / GT-lookup work
- **PRE-EXISTING** — pre-dates this session, verified during audit
- **DEFERRED** — flagged earlier and explicitly left for later

Statuses:
- ✅ **DONE** — fixed in this session
- ⊘ **IGNORED** — user has explicitly decided not to fix
- (no status) — open / pending decision

---

## CRITICAL — breaks correctness or persistence

### Bug 1. `clear_db.py` doesn't reset the two new tables
**File:** [db/clear_db.py:68-76](db/clear_db.py)
**Tag:** NEW
**Status:** ✅ DONE (2026-06-18)

The `tables` list omits `autorag_strategy_counters` and `autorag_runtime_flags`.
Running `python db/clear_db.py --logs --confirm` wipes QueryLog/Events/Reports
but leaves `s1_dynamic_k.success_count = 5` and `dynamic_k_promoted = True` in
place — so a "fresh" test starts with the old promotion already active and the
cascade still skips S1.

**Fix:** add the two table names to the list.

```python
tables = [
    "autorag_repair_reports",
    "autorag_low_recall_events",
    "autorag_query_log",
    "autorag_eval_snapshots",
    "autorag_pipeline_config",
    "autorag_chunk_snapshots",
    "autorag_adaptation_log",
    "autorag_strategy_counters",   # ADD
    "autorag_runtime_flags",       # ADD
]
```

---

### Bug 2. `del_query.py` leaves FK orphans
**File:** [db/del_query.py:23-28](db/del_query.py)
**Tag:** PRE-EXISTING
**Status:** ✅ DONE (2026-06-18)

When deleting a QueryLog and its LowRecallEvents, the script does NOT delete
the associated `RepairReport`, `AdaptationLog`, or `ChunkSnapshot` rows that
reference those events. SQLite doesn't enforce FK constraints by default →
silent orphan rows accumulate.

**Fix:** delete the child rows for each event before deleting the event.

```python
from db.models import (
    QueryLog, LowRecallEvent, RepairReport, AdaptationLog, ChunkSnapshot,
)

events = session.query(LowRecallEvent).filter(
    LowRecallEvent.query_log_id == log.id
).all()
for event in events:
    session.query(RepairReport).filter(
        RepairReport.event_id == event.id
    ).delete()
    session.query(AdaptationLog).filter(
        AdaptationLog.event_id == event.id
    ).delete()
    session.query(ChunkSnapshot).filter(
        ChunkSnapshot.event_id == event.id
    ).delete()
    session.delete(event)
session.delete(log)
session.commit()
```

---

## HIGH — real flaws with workarounds

### Bug 3. Cascade ignores GT even though gt_lookup has it
**File:** [repair/orchestrator.py:344-360](repair/orchestrator.py)
**Tag:** NEW
**Status:** ✅ DONE (2026-06-18)

`_get_ground_truths_for_query` returns `[]` unconditionally. Now that
`gt_lookup` provides real ground truths for any dataset-matched query, the
cascade could pass them through `_probe_metrics` → `_is_improved` would judge
on precision/recall/accuracy instead of relying solely on `top1_score`. Much
stronger signal for win-decisions.

**Fix:** look up GT via the new module, fall back to `[]` if unavailable.

```python
def _get_ground_truths_for_query(query_log) -> list:
    """Returns dataset-matched ground truths if available, else []."""
    try:
        from controllers.gt_lookup import lookup_ground_truth
        gts = lookup_ground_truth(query_log.query)
        return gts if gts else []
    except Exception:
        return []
```

---

### Bug 4. Dashboard detection-rule matrix is stale
**File:** [dashboard/app.py:699-702](dashboard/app.py)
**Tag:** NEW
**Status:** ✅ DONE (2026-06-18)

The displayed matrix shows `semantic_mismatch` threshold as `0.70` absolute.
The actual rule is now `COHERENCE_RATIO = 0.65 × top1_score` (relative). The
user-facing detection docs lie about how the system works.

**Fix:** update the row.

```html
<tr>
    <td><code>semantic_mismatch</code></td>
    <td>Mean pairwise chunk sim below ratio × top1 (K-adaptive)</td>
    <td class="thresh">0.65 × top1</td>
</tr>
```

---

### Bug 5. Dashboard truthy check hides `ctx_q_sim = 0.0`
**File:** [dashboard/app.py:1000](dashboard/app.py), [dashboard/app.py:1006](dashboard/app.py)
**Tag:** PRE-EXISTING
**Status:** ✅ DONE (2026-06-18)

```python
col1.metric("CTX↔QUERY SIM",  f"{cqs:.4f}" if cqs else "N/A")
```

`if cqs` is False for `0.0`. A genuinely-failed retrieval with zero context
similarity displays "N/A", hiding the worst data — making it look like the
metric wasn't computed at all.

**Fix:** explicit None check.

```python
col1.metric("CTX↔QUERY SIM",  f"{cqs:.4f}" if cqs is not None else "N/A")
```

Apply to both call sites (lines 1000 and 1006).

---

## MEDIUM — semantic issues, footguns

### Bug 6. `update_log_eval_metrics` writes to a non-existent column
**File:** [logger/query_logger.py:60-65](logger/query_logger.py)
**Tag:** PRE-EXISTING (now hit from the /query path too)
**Status:** ✅ DONE (2026-06-18) — param dropped, callers in evaluation.py and gt_lookup.py updated

```python
log_entry.retrieved_contexts = json.dumps(clean_contexts, ensure_ascii=False)
```

`QueryLog` has `retrieved_chunks`, not `retrieved_contexts`. SQLAlchemy
silently drops the write (it only persists mapped columns). No data is lost
(chunks are stored upstream via `log_query`'s `retrieved_chunks` field), but
the call is dead code and misleading.

**Fix:** drop the `retrieved_contexts` parameter from `update_log_eval_metrics`
entirely. Callers (`controllers/evaluation.py`, `controllers/gt_lookup.py`)
should stop passing it.

```python
def update_log_eval_metrics(log_id: int, answer_sem_sim: float, ctx_q_sim: float):
    session = get_session()
    try:
        log_entry = session.query(QueryLog).filter(QueryLog.id == log_id).first()
        if log_entry:
            log_entry.answer_sem_sim = answer_sem_sim
            log_entry.ctx_q_sim = ctx_q_sim
            session.commit()
    except Exception as e:
        session.rollback()
        print(f"      [db] failed to update metrics: {e}")
    finally:
        session.close()
```

Then in `gt_lookup.enrich_log_with_gt` and `controllers/evaluation.py`, drop
the `retrieved_contexts=...` kwarg from the call.

---

### Bug 7. GT enrichment doubles user-visible latency
**File:** [controllers/retrieval.py:109-122](controllers/retrieval.py)
**Tag:** NEW
**Status:** ✅ DONE (2026-06-18)

Fixed as part of the A+B+C pipeline-latency pass. `_post_process_log` runs both
GT enrichment and `run_detectors` on a daemon thread; HTTP response returns
immediately after `log_query`.


`enrich_log_with_gt` runs ~6 synchronous embedding calls (~1–2s) on
dataset-matched queries, blocking the HTTP response. Combined with the
detector's synchronous run, matched queries can hit 4–8s response time.

**Fix (lightweight):** move both enrichment and `run_detectors` to a
background thread.

```python
import threading

def _enrich_async(log_id, query, answer, contexts, gts):
    try:
        from controllers.gt_lookup import enrich_log_with_gt
        enrich_log_with_gt(log_id, query, answer, contexts, gts)
    except Exception as e:
        logging.warning(f"[retrieval] async GT enrichment failed: {e}")

# In answer_query, after log_query:
if log_id > 0:
    try:
        from controllers.gt_lookup import lookup_ground_truth
        gts = lookup_ground_truth(query)
        if gts:
            threading.Thread(
                target=_enrich_async,
                args=(log_id, query, response["answer"],
                      [d.page_content for d in docs], gts),
                daemon=True,
            ).start()
    except Exception as e:
        logging.warning(f"[retrieval] GT lookup failed: {e}")

# Same for run_detectors:
threading.Thread(target=run_detectors, args=(log_id,), daemon=True).start()
```

Trade-off: if the user reads the dashboard within ~2s of the response, GT-backed
metrics may not yet be visible. Acceptable for most workflows.

---

### Bug 8. Dataset path is hardcoded
**File:** [controllers/gt_lookup.py:24-27](controllers/gt_lookup.py)
**Tag:** NEW
**Status:** ✅ DONE (2026-06-18) — `settings.gt_dataset_path`

`_DATASET_PATH` only points to `long_ans.json`. Switching to `Qun_Ans1.json`
requires editing source.

**Fix:** read from `config.py` with a sensible default.

```python
# config.py
class Settings(BaseSettings):
    ...
    gt_dataset_path: str = "dataset/long_ans.json"

# gt_lookup.py
from config import settings
_DATASET_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 settings.gt_dataset_path)
)
```

---

### Bug 9. Duplicate questions overwrite each other in the GT index
**File:** [controllers/gt_lookup.py:55](controllers/gt_lookup.py)
**Tag:** NEW
**Status:** ✅ DONE (2026-06-18) — merge + dedupe

`_gt_index[norm_q] = a` — last entry wins. SQuAD-style datasets repeat the
same `qun` across `para` IDs. Usually harmless (same `ans`) but worth flagging.

**Fix:** merge into a deduplicated list of answers.

```python
for item in data:
    q = item.get("qun", "")
    a = item.get("ans", [])
    if not (q and a):
        continue
    key = _normalize(q)
    answers = a if isinstance(a, list) else [a]
    existing = _gt_index.get(key, [])
    # Dedupe while preserving order
    merged = list(dict.fromkeys(existing + answers))
    _gt_index[key] = merged
```

---

### Bug 10. Dashboard leaks DB sessions
**File:** [dashboard/app.py:591](dashboard/app.py)
**Tag:** PRE-EXISTING
**Status:** ✅ DONE (2026-06-18) — `@st.cache_resource`

`session = get_session()` runs at the top of every Streamlit script execution.
Streamlit re-runs the entire script on each interaction → one session leak per
click. SQLite SessionLocal isn't bounded, so this accumulates.

**Fix:** scope the session via Streamlit's caching or close it at the end.

```python
# Option A (simplest): use streamlit's resource cache
@st.cache_resource
def _get_dashboard_session():
    return get_session()

session = _get_dashboard_session()

# Option B: explicit close at script tail (works less cleanly with Streamlit's
# script-rerun model — A is preferred).
```

---

### Bug 11. Inconsistent `log_id` guards
**Files:** [controllers/retrieval.py:109](controllers/retrieval.py), [detector/detectors.py:124](detector/detectors.py)
**Tag:** NEW
**Status:** ✅ DONE (2026-06-18) — detectors.py now uses `<= 0`

`retrieval.py` uses `if log_id > 0`; `detectors.py` uses `if log_id < 0: return`.
`log_id == 0` is impossible in practice (SQLite autoincrement starts at 1), but
the asymmetry is sloppy.

**Fix:** standardize on `>= 0` everywhere (covers logger's `-1` sentinel).

```python
# retrieval.py
if log_id >= 0:
    try:
        ...
```

Or change the logger's sentinel from `-1` to `None` and use `is not None`
checks everywhere. Either is fine — pick one rule.

---

## LOW — dead code, hazardous scripts, deferred items

### Bug 12. `embed_documents.py` and `rag_app.py` are dimensional-mismatch landmines
**Files:** [unused/embed_documents.py](unused/embed_documents.py), [unused/rag_app.py](unused/rag_app.py)
**Tag:** PRE-EXISTING
**Status:** ✅ DONE (2026-06-18) — moved to `unused/` with README warning

Both use HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (**384-dim**) and
target a Pinecone index named `rag-index`. The main pipeline uses Ollama
`mxbai-embed-large` (**1024-dim**) on a different index. If a new developer
runs either:
- Wrong-dimension vectors get uploaded → Pinecone insertion fails OR succeeds
  silently with mismatched dim
- Different chunk params (500/80 vs. 1250/200) → wrong chunking
- Hardcoded index name → potentially writes to the wrong index entirely

**Fix:** delete both files.

```bash
rm embed_documents.py rag_app.py
```

If you want a "minimal RAG demo" reference, move them into a `legacy/` folder
and add a README warning they're incompatible with the production pipeline.

---

### Bug 13. Retrieval-gate 2-word heuristic misroutes topical queries
**File:** [organiser/retrieval_gate.py:49-55](organiser/retrieval_gate.py)
**Tag:** DEFERRED
**Status:** ⊘ IGNORED (per user, 2026-06-18)

Short queries like "binary search", "elon musk", "transformer architecture"
get routed to DIRECT — bypassing Pinecone, answered from LLM general
knowledge instead of the user's docs.

**Fix:** delete lines 49–55. Tier 1 exact-match catches real chitchat
already; Tier 2 LLM classifier handles the ambiguous middle.

---

### Bug 14. Retrieval-gate Tier 2 doubles latency on ambiguous queries
**File:** [organiser/retrieval_gate.py:58-70](organiser/retrieval_gate.py)
**Tag:** DEFERRED
**Status:** ⊘ IGNORED (per user, 2026-06-18)

Every ambiguous query → one mistral classification + one mistral answer
generation. ~2× LLM round-trips.

**Fix:** cache classification results by query string, or replace Tier 2
with deterministic regex-based question detection (question word at start +
length > N).

```python
from functools import lru_cache

@lru_cache(maxsize=1024)
def _classify_cached(query: str) -> bool:
    # ... existing Tier 2 LLM call returning bool
```

---

### Bug 15. Gate fallback dict missing `gate_detail` key
**File:** [organiser/retrieval_gate.py:65-67](organiser/retrieval_gate.py)
**Tag:** DEFERRED
**Status:** ✅ DONE (2026-06-18) — added in both gate fallback paths

```python
gate_result = {"needs_retrieval": True, "reason": "fallback"}
```

No `gate_detail`. Currently no caller reads it on this path, but a future one
might `KeyError`.

**Fix:** add the key.

```python
gate_result = {
    "needs_retrieval": True,
    "reason": "fallback",
    "gate_detail": "Gate failed; defaulting to retrieval",
}
```

---

### Bug 16. `add_chunks/pipeline.py` crashes on tiny input
**File:** [add_chunks/pipeline.py:58-63](add_chunks/pipeline.py)
**Tag:** DEFERRED
**Status:** ✅ DONE (2026-06-18)

```python
sizes = [len(c.page_content) for c in chunks]
print(f"[add-chunks] recursive | {len(chunks)} chunks | sizes: {min(sizes)}-{max(sizes)} chars")
```

If `_enforce_min_chunk_size` reduces output to 0 chunks, `min(sizes)` raises
`ValueError: min() arg is an empty sequence`.

**Fix:** guard.

```python
if not chunks:
    print(f"[add-chunks] recursive | 0 chunks (input too short)")
    return chunks
sizes = [len(c.page_content) for c in chunks]
print(f"[add-chunks] recursive | {len(chunks)} chunks | sizes: {min(sizes)}-{max(sizes)} chars")
```

---

### Bug 17. `/add-chunks` endpoint duplicates `add_chunks.add_chunk`
**File:** [api/routes.py:46-89](api/routes.py)
**Tag:** DEFERRED
**Status:** ✅ DONE (2026-06-18) — endpoint now delegates to `add_chunk()`

The endpoint reimplements the splitter + min-size enforcement instead of
calling `add_chunks.add_chunk(req.text, req.source)`. Two places to update
when the chunking rule changes.

**Fix:** delegate.

```python
@router.post("/add-chunks")
async def add_chunks_endpoint(req: AddChunksReq):
    from add_chunks import add_chunk
    chunks = add_chunk(req.text, req.source)
    sizes = [len(c.page_content) for c in chunks]
    result = {
        "strategy": "recursive",
        "num_chunks": len(chunks),
        "size_range": f"{min(sizes)}-{max(sizes)}" if sizes else "0",
        "chunks": [
            {"content": c.page_content[:300], "chars": len(c.page_content)}
            for c in chunks
        ],
    }
    if req.ingest:
        from services.llm_factory import get_vector_store
        vs = get_vector_store(req.namespace)
        vs.add_documents(chunks)
        result["ingested"] = True
        result["namespace"] = req.namespace or "default"
    return result
```

---

### Bug 18. `_direct_answer` skips `run_detectors`
**File:** [controllers/retrieval.py:203](controllers/retrieval.py)
**Tag:** DEFERRED
**Status:** ⊘ IGNORED (per user, 2026-06-18)

Gate-routed direct answers never run through detection. Combined with Bug 13
(2-word topical queries misrouted to DIRECT), a hallucinated direct answer on
a topical query like "binary search" would never get flagged or repaired.

**Fix:** decide policy — either (a) accept that direct answers are
hallucination-prone and surface a warning in the dashboard, or (b) run a
lightweight `llm_uncertainty` check even on direct answers.

```python
# Minimal: at least flag obvious refusals
from detector.detectors import UNCERTAINTY_PHRASES
lower_ans = direct_response.lower()
if any(p in lower_ans for p in UNCERTAINTY_PHRASES):
    # Could write a special LowRecallEvent here
    logging.info(f"[direct] uncertainty detected: log_id={log_id}")
```

---

## NIT — cleanup with no functional impact

### Bug 19. `_DATASET_PATH` not normalized
**File:** [controllers/gt_lookup.py:24-27](controllers/gt_lookup.py)
**Tag:** NEW
**Status:** ✅ DONE (2026-06-18)

The joined path contains a `..` segment. Works on all OSes but looks ugly in
error messages.

**Fix:** wrap in `os.path.normpath(...)`.

```python
_DATASET_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "dataset", "long_ans.json",
))
```

(Combine with Bug 8's fix if you do them together.)

---

### Bug 20. `_gt_index` first-load race
**File:** [controllers/gt_lookup.py:42-65](controllers/gt_lookup.py)
**Tag:** NEW
**Status:** ✅ DONE (2026-06-18) — `threading.Lock` with double-checked init

Two concurrent first requests could both enter `_load_dataset`. Idempotent
(same keys overwrite with same values) but theoretically wasteful.

**Fix:** thread lock.

```python
import threading
_load_lock = threading.Lock()

def _load_dataset():
    global _gt_index, _loaded
    if _loaded:
        return
    with _load_lock:
        if _loaded:
            return
        # ... existing load logic
```

---

### Bug 21. `run_full_refresh` writes weird `EvalSnapshot` rows
**File:** [auto_indexer/engine.py:417-426](auto_indexer/engine.py)
**Tag:** PRE-EXISTING
**Status:** ✅ DONE (2026-06-18) — block removed

```python
snap = EvalSnapshot(
    namespace=self.namespace,
    llm="auto-indexer",
    rouge_l=0.0,
    sem_sim=0.0,
    ctx_q_sim=staleness["avg_drift"],   # semantically wrong column
    ctx_gt_sim=float(staleness["stale_count"]),
)
```

`avg_drift` jammed into `ctx_q_sim`, `stale_count` jammed into `ctx_gt_sim`.
Confuses dashboard eval-history view; doesn't break anything.

**Fix:** either (a) skip the persistence (these aren't evaluations) or (b)
add a dedicated `IndexHealthSnapshot` table.

```python
# Cheapest: just delete the try block entirely. Index-refresh telemetry
# belongs in its own endpoint, not the eval-history feed.
```

---

### Bug 22. `auto_worker.py` heartbeat uses `\r`
**File:** [auto_worker.py:112-126](auto_worker.py)
**Tag:** PRE-EXISTING
**Status:** ✅ DONE (2026-06-18) — heartbeat now uses normal newline

Heartbeat `print(..., end="\r")` collides with cascade `print(..., end="\n")`
output, producing garbled lines when both run.

**Fix:** use a single status line via `logging`, or just drop the `\r` and
let it scroll.

---

### Bug 23. `config.py::cooldown_seconds` deprecated but kept
**File:** [config.py:26](config.py)
**Tag:** PRE-EXISTING
**Status:** ✅ DONE (2026-06-18) — line removed

Cooldown is a relic of the pre-cascade retry loop. The setting still loads
from `.env` but isn't read anywhere.

**Fix:** remove the line. (Marker that the cascade is the single source of
repair-execution truth.)

---

## Final status

**All 23 bugs are either resolved or explicitly ignored.**

| Status | Count | Bugs |
|---|---|---|
| ✅ DONE | 20 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 19, 20, 21, 22, 23 |
| ⊘ IGNORED (per user) | 3 | 13, 14, 18 |
| Open | 0 | — |

### Resolved this session

| Bug | What changed |
|---|---|
| 1 | `clear_db.py` now includes `autorag_strategy_counters` and `autorag_runtime_flags` |
| 2 | `del_query.py` cascade-deletes `RepairReport`, `AdaptationLog`, `ChunkSnapshot` before events |
| 3 | `_get_ground_truths_for_query` reads from `gt_lookup` — cascade gets real GT signals |
| 4 | Dashboard rule matrix shows `0.65 × top1` relative threshold for `semantic_mismatch` |
| 5 | Dashboard `ctx_q_sim` rendering uses `is not None`, no longer hides 0.0 |
| 6 | `update_log_eval_metrics` dropped the `retrieved_contexts` param + write to dead column |
| 7 | `_post_process_log` daemon thread runs GT enrichment + detectors after response returns |
| 8 | `settings.gt_dataset_path` (config.py) drives the path; defaults to `dataset/long_ans.json` |
| 9 | Duplicate questions merged + deduped instead of overwritten |
| 10 | Dashboard session lifted into `@st.cache_resource` — no per-click leak |
| 11 | Detector log_id guard standardized to `<= 0` |
| 12 | `embed_documents.py`, `rag_app.py` moved to `unused/` with README warning |
| 15 | Gate fallback dicts (both paths) include `gate_detail` key |
| 16 | `add_chunks/pipeline.py` guards empty chunks before min/max |
| 17 | `/add-chunks` endpoint delegates to `add_chunk()` — single source of chunking logic |
| 19 | `_DATASET_PATH` is `os.path.normpath`-resolved |
| 20 | `_load_dataset` is thread-safe via `threading.Lock` with double-checked init |
| 21 | `run_full_refresh` no longer pollutes `EvalSnapshot` with index-health data |
| 22 | `auto_worker.py` heartbeat uses normal newline — no `\r` collision with cascade prints |
| 23 | `cooldown_seconds` removed from `config.py` |

### Pipeline optimizations (not in the bug list, applied alongside)

- A+D — dropped duplicate Pinecone search in `answer_query` and `generate_answer_only`
- B — batched detector embeddings (`embed_documents([...])` instead of N `embed_query` calls)

Combined with Bug 7 (threaded enrichment + detection), expected user-visible
latency: **~4.4s → ~3.0s**.

---

*Last updated: 2026-06-18*
