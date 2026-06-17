# CLAUDE.md — Implementation Plan: Ordered Repair Cascade + Dynamic-Retrieval Promotion

This file is a forward-looking implementation plan, not documentation of current
behavior. It describes the changes to the existing Self-Organising RAG pipeline
required to satisfy the new spec (ordered 4-strategy single-pass repair, 30%/≥5
trigger, and Strategy-1 promotion to the main pipeline).

---

## 0. Open Questions (please confirm before I start coding)

| # | Question | My default if you say nothing |
|---|---|---|
| Q1 | Trigger denominator: "30% of total queries flagged" — lifetime `QueryLog` rows, or a rolling window? | **Lifetime**, constants in `auto_worker.py` make the window swappable later |
| Q2 | Reuse existing `LowRecallEvent.unfixable` as the "failed all 4 strategies" flag, or add a new `unresolved` column? | **Reuse `unfixable=True`**. Cleaner schema, same meaning |
| Q3 | Strategy 4 alternate LLM: `services.llm_factory.get_fallback_llm()` → `gemma3:27b`. Use that? | **Yes** |
| Q4 | Strategy 2 chunk-size source: reuse `detector/decision_engine.diagnose()` to pick the size variant, with K fixed at 5? | **Yes** |
| Q5 | Delete the legacy in-orchestrator cascade (query-reformulation + llm-fallback + auto-resolve) inside `handle_event()`? | **Yes** — the new orchestrator owns the cascade. Keeping both would double-fire Strategy 4 |
| Q6 | Promotion direction: once Strategy 1 is promoted to main pipeline, never demote — even if it later regresses? | **Yes, one-way promotion** |
| Q7 | Counter scope: per-namespace or global? | **Global** (single row per strategy). Easy to upgrade to per-namespace later |
| Q8 | `RepairReport.strategy_used` is `String(50)` — okay to write values `s1_dynamic_k`, `s2_chunk_size`, `s3_combined`, `s4_alt_llm`? | **Yes** |

---

## 1. Current State Snapshot (so the diff is explicit)

These are the pieces the plan touches. File paths are clickable.

- Per-query path: [controllers/retrieval.py:10](controllers/retrieval.py) — `answer_query()`, hardcoded `k=5`, calls `run_detectors()` after logging.
- Detector: [detector/detectors.py:143](detector/detectors.py) — `run_detectors()` writes a `LowRecallEvent` and sets `QueryLog.flagged = True`.
- Background loop: [auto_worker.py:28](auto_worker.py) — polls every 5s, fires repair once `BATCH_THRESHOLD = 5` unresolved events accumulate. Already excludes `unfixable`.
- Strategy selection: [detector/decision_engine.py](detector/decision_engine.py) — `diagnose()` + `select_strategy()` choose ONE strategy per attempt and re-queue with cooldown. **This is the cascade we're replacing.**
- Repair execution: [repair/orchestrator.py:358](repair/orchestrator.py) — `handle_event()` does dynamic-K + rechunk + probe + rollback + a hidden secondary cascade (reformulation → fallback LLM). **The hidden cascade gets removed; the orchestrator becomes one strategy step.**
- Dynamic-K helper: [repair/orchestrator.py:58](repair/orchestrator.py) — `_dynamic_k_selection()` already maps `question_category → K bounds`. Reused as-is for Strategy 1.
- Rechunk strategies: [repair/chunker.py](repair/chunker.py) — `rechunk_semantic / rechunk_llm / rechunk_entropy`. Reused.
- Snapshot + rollback: [repair/reembedder.py](repair/reembedder.py) — `reembed()` snapshots before delete; `rollback_from_snapshot()` restores. Reused.
- Resolution judge: [repair/orchestrator.py:247](repair/orchestrator.py) — `_is_improved()` is the composite check (precision/recall/accuracy + non-answer detection). **Reused unchanged as the "resolved?" oracle for every strategy.**
- Alternate LLM: [services/llm_factory.py:76](services/llm_factory.py) — `get_fallback_llm()` returns `gemma3:27b`. Strategy 4 uses this.
- Question classifier: [controllers/metrics.py:314](controllers/metrics.py) — `classify_question()` returns `short_factual | complex | cross_section`. Strategy 1 uses this.

---

## 2. New / Changed Files (summary)

| File | Change | Risk |
|---|---|---|
| `db/models.py` | **+** add `StrategyCounter` table; **+** add `RuntimeFlag` table (for promotion gate) | Low — additive, `init_db()` handles create-all |
| `auto_worker.py` | Replace polling logic: count-based trigger (30% AND ≥5 pending), call new orchestrator | Medium — controls when repair fires |
| `repair/cascade.py` | **NEW** — single-pass ordered cascade: `run_repair_cascade(event_id)` | High — the new heart of the system |
| `repair/orchestrator.py` | Strip the legacy in-function cascade; keep only "apply one rechunk + probe" as a primitive callable from `cascade.py` | High — careful refactor |
| `controllers/retrieval.py` | After Strategy-1 promotion: switch hardcoded `k=5` → `_dynamic_k_for_query()` when the runtime flag is set | Medium — affects main user path |
| `detector/decision_engine.py` | Keep `diagnose()` and `STRATEGY_CONFIGS`; deprecate (do not delete) `select_strategy()` and cooldown logic | Low |
| `repair/llm_swap.py` | **NEW** — Strategy 4 helper: re-run answer with fallback LLM on existing chunks; judge with `_is_improved()` | Low |
| `api/routes.py` | **+** read-only endpoints `/strategy-counters` and `/runtime-flags` for the dashboard | Low |

I will **not** rewrite the dashboard. It already reads `RepairReport.strategy_used`, which keeps working.

---

## 3. Data Model Changes

### 3a. `StrategyCounter` (new)
Persists the success counters. Mirrors the existing SQLite pattern (`db/models.py`).

```python
class StrategyCounter(Base):
    __tablename__ = "autorag_strategy_counters"
    id          = Column(Integer, primary_key=True)
    strategy    = Column(String(50), unique=True)   # s1_dynamic_k | s2_chunk_size | s3_combined | s4_alt_llm
    success_count = Column(Integer, default=0)
    last_incremented_at = Column(DateTime, nullable=True)
```

Why a table, not a JSON file or in-memory dict: the dashboard reads from
SQLite, the auto-worker writes to SQLite, and the FastAPI process reads on
startup. Three processes, one shared truth — same pattern as every other
table here. In-memory would desync between `uvicorn` and `auto_worker.py`.

### 3b. `RuntimeFlag` (new)
Boolean flags for one-way promotions.

```python
class RuntimeFlag(Base):
    __tablename__ = "autorag_runtime_flags"
    id    = Column(Integer, primary_key=True)
    name  = Column(String(80), unique=True)   # e.g. "dynamic_k_promoted"
    value = Column(Boolean, default=False)
    set_at = Column(DateTime, default=datetime.utcnow)
```

`controllers/retrieval.answer_query()` reads `dynamic_k_promoted` once per
request (cheap SQLite read — same session pattern as everything else).
The auto-worker sets it to `True` when `s1.success_count >= 5`.

### 3c. Reuse existing fields
- `LowRecallEvent.unfixable = True` → "failed all 4 strategies, do not retry."
- `LowRecallEvent.resolved = True` → "one of the strategies succeeded."
- `LowRecallEvent.attempts` → we set this to 1 once per event (single pass).
- `RepairReport.strategy_used` → exact strategy name that resolved it
  (`s1_dynamic_k` / `s2_chunk_size` / `s3_combined` / `s4_alt_llm` /
  `none` if all four failed).

No schema migrations needed for these — `init_db()` create-all handles new
tables; existing rows keep working.

---

## 4. Repair Trigger (auto_worker.py)

### Required behavior
Repair fires when **both**:
- Pending count ≥ 5
- Pending count / total `QueryLog` count ≥ 0.30

"Pending" = `LowRecallEvent.resolved = False AND unfixable = False`.

### Implementation
Replace the `current_queue_count >= BATCH_THRESHOLD` block in
`auto_worker.py::run_batch_worker()` with:

```python
PENDING_MIN = 5
PENDING_RATIO = 0.30

total_queries = session.query(QueryLog).count()
pending_events = (
    session.query(LowRecallEvent)
    .filter(LowRecallEvent.resolved == False, LowRecallEvent.unfixable == False)
    .order_by(LowRecallEvent.timestamp.asc())
    .all()
)
pending_count = len(pending_events)

ratio_ok = total_queries > 0 and (pending_count / total_queries) >= PENDING_RATIO
count_ok = pending_count >= PENDING_MIN

if not (ratio_ok and count_ok):
    # heartbeat log, sleep, continue
    ...
    continue

# Trigger — process EVERY pending event in this batch (not a fixed slice of 5)
for event in pending_events:
    from repair.cascade import run_repair_cascade
    run_repair_cascade(event.id)
```

Notes:
- Processing all pending events (not just the first 5) avoids a starvation
  bug where the queue stays above-threshold forever because events 6+ never
  get touched.
- Old `MAX_ATTEMPTS = 5` retry / cooldown logic is gone. Each event gets
  exactly one cascade pass; `unfixable` is the terminal state.
- The legacy `select_strategy` / `check_cooldown` / `set_cooldown` calls go
  away from `auto_worker.py`.

---

## 5. Repair Cascade — `repair/cascade.py` (NEW)

Single entry point, ordered, single-pass, with rollback.

```python
def run_repair_cascade(event_id: int) -> dict:
    """
    Single-pass ordered cascade.
    Each strategy is tried; the first one that satisfies _is_improved()
    wins, increments its counter, writes a RepairReport, and returns.
    All four failing → mark unfixable=True, RepairReport(strategy_used="none").
    """
```

### Strategy order (from spec)
1. `s1_dynamic_k` — vary retrieval K by question category; chunk content unchanged.
2. `s2_chunk_size` — vary chunk size in Pinecone; K fixed at 5.
3. `s3_combined` — dynamic K + varied chunk size together.
4. `s4_alt_llm` — same chunks + K, swap LLM for `gemma3:27b`.

### Pseudocode

```python
def run_repair_cascade(event_id):
    session = get_session()
    event, log = load(event_id)
    if event.resolved or event.unfixable:
        return
    event.attempts = 1   # single pass; field still useful for dashboards

    snapshot_taken = False
    metrics_before = probe_metrics(log.query, k=5)   # baseline K=5 (or _dynamic_k if promoted)

    # ── Possibly skip S1 if it's been promoted to main pipeline ──
    skip_s1 = is_flag_set("dynamic_k_promoted")

    cascade = []
    if not skip_s1:
        cascade.append(("s1_dynamic_k", run_s1_dynamic_k))
    cascade.extend([
        ("s2_chunk_size",   run_s2_chunk_size),
        ("s3_combined",     run_s3_combined),
        ("s4_alt_llm",      run_s4_alt_llm),
    ])

    diagnosis = diagnose(event, log)   # reused from decision_engine
    winning_strategy = None
    metrics_after = metrics_before

    for name, fn in cascade:
        result = fn(event, log, diagnosis, metrics_before)
        if _is_improved(metrics_before, result["metrics_after"]):
            winning_strategy = name
            metrics_after = result["metrics_after"]
            break
        # Strategy did not resolve → roll back any Pinecone change it made
        if result.get("pinecone_touched"):
            rollback_from_snapshot(event_id, new_chunk_ids=result["new_chunk_ids"])

    if winning_strategy:
        event.resolved = True
        increment_counter(winning_strategy)        # +1 to StrategyCounter.success_count
        if winning_strategy == "s1_dynamic_k":
            maybe_promote_dynamic_k()              # see §7
        write_repair_report(event, winning_strategy, metrics_before, metrics_after)
    else:
        event.unfixable = True
        write_repair_report(event, "none", metrics_before, metrics_after)

    session.commit()
```

### Per-strategy contract (each `fn` returns the same shape)

```python
{
    "metrics_after": {top1_score, context_precision, recall, answer_accuracy, answer, chunks, ...},
    "pinecone_touched": bool,        # True if we replaced vectors and need rollback on fail
    "new_chunk_ids": list[str],      # only when pinecone_touched
    "details": dict,                 # for provenance — k_used, chunk_size, llm_used, ...
}
```

The same `_probe_metrics()` from `repair/orchestrator.py` builds `metrics_after`.
The same `_is_improved()` is the universal judge.

### Strategy 1 — `run_s1_dynamic_k`
- `category = classify_question(log.query)` (or `log.question_category`).
- `k = _dynamic_k_selection(log.query, category, quick_scores)` — already exists.
- **No Pinecone change.** Just re-probe at the new K.
- `pinecone_touched = False`.

### Strategy 2 — `run_s2_chunk_size`
- Fix K = 5.
- Use `diagnose()` output to pick `STRATEGY_CONFIGS[recommended_strategy]`:
  `reduce_chunk_size` / `increase_chunk_size` / `tighten_chunks` /
  `large_coherent_chunks` / `re_ingest`. These already exist in
  `decision_engine.py`.
- Pull the failing chunks via `_get_chunk_ids_for_query(log.query, k=5)`,
  rechunk with `rechunk_semantic(text, source, chunk_size, chunk_overlap)`.
- `reembed(..., old_chunk_ids, event_id)` — this also writes the
  `ChunkSnapshot` for rollback.
- Probe at K = 5.
- `pinecone_touched = True`.

### Strategy 3 — `run_s3_combined`
- Pick K via the same dynamic-K selection.
- Pick chunk size via the same `diagnose()` recommendation.
- Same rechunk + reembed + probe path as Strategy 2, but K from Strategy 1.
- `pinecone_touched = True`.

Edge case: if Strategy 2 just touched Pinecone and was rolled back, Strategy
3 starts from a clean baseline — the snapshot restore brings the index back
to pre-S2 state. We must `rollback_from_snapshot` **before** S3 calls
`reembed` so the snapshot table doesn't keep stale entries.

### Strategy 4 — `run_s4_alt_llm` (`repair/llm_swap.py`)
- No Pinecone change.
- Retrieve chunks with current K (5 if not promoted, dynamic if promoted).
- Build the same prompt template from `controllers/retrieval.answer_query()`
  but invoke `get_fallback_llm()` (`gemma3:27b`).
- Replace `answer` in the probe result; reuse `_is_improved()` — the
  non-answer check catches "still can't answer."
- `pinecone_touched = False`.

### Rollback guarantee
Pinecone snapshot/rollback only matters for S2 and S3. The cascade explicitly
rolls back any failed S2/S3 attempt **before** moving on to the next
strategy. The "previous retrieval + chunk config" is the original ingestion
state captured in `ChunkSnapshot` rows keyed by `event_id`.

---

## 6. Counter & Promotion Logic

### Increment
`increment_counter(strategy_name)` does a single `UPDATE
autorag_strategy_counters SET success_count = success_count + 1,
last_incremented_at = now() WHERE strategy = :name`. If the row doesn't
exist, INSERT it.

### Promotion
```python
PROMOTION_THRESHOLD = 5

def maybe_promote_dynamic_k():
    s1 = get_counter("s1_dynamic_k")
    if s1.success_count >= PROMOTION_THRESHOLD and not is_flag_set("dynamic_k_promoted"):
        set_flag("dynamic_k_promoted", True)
        logging.info("[promotion] Strategy 1 promoted to main pipeline.")
```

### Main-pipeline switch — `controllers/retrieval.py`
Replace:
```python
docs_with_scores = vector_store.similarity_search_with_score(query, k=5)
```
with:
```python
k = _resolve_main_k(query)   # helper that reads RuntimeFlag once per request
docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
```

```python
def _resolve_main_k(query: str) -> int:
    from db.session import get_session
    from db.models import RuntimeFlag

    s = get_session()
    try:
        promoted = (
            s.query(RuntimeFlag)
            .filter(RuntimeFlag.name == "dynamic_k_promoted", RuntimeFlag.value == True)
            .first()
        )
    finally:
        s.close()

    if not promoted:
        return 5

    # Promoted path — classify and pick dynamic K
    from controllers.metrics import classify_question
    from repair.orchestrator import _dynamic_k_selection
    category = classify_question(query)
    return _dynamic_k_selection(query, category, scores=None)  # no scores yet pre-retrieval
```

Note: `_dynamic_k_selection` already gracefully degrades when `scores` is
`None` — it returns the category midpoint (see [repair/orchestrator.py:81](repair/orchestrator.py)).

### Skip-S1-after-promotion
Cascade reads the same flag before building its strategy list. Test plan
below covers both branches.

---

## 7. Detector / Decision-Engine Cleanup

Keep `diagnose()` (Strategy 2/3 need it). **Deprecate but do not delete**:
- `select_strategy()` — no longer used; mark deprecated in a one-line module docstring at the top.
- `check_cooldown()`, `set_cooldown()`, `LowRecallEvent.cooldown_until`/`last_repair_at` — single-pass cascade has no cooldown concept. Mark deprecated; remove call sites; leave columns NULL.

Rationale for leaving the columns: avoiding a destructive migration. The
project explicitly uses `init_db()` create-all, not Alembic.

---

## 8. Provenance (still written, simpler shape)

For each cascade run we write:
- **One `RepairReport`** with `strategy_used` = winning strategy or `"none"`,
  plus `score_before / score_after / precision_before / precision_after /
  recall_before / recall_after / accuracy_before / accuracy_after /
  chunks_before_text / chunks_after_text / dynamic_k`. Same columns the
  dashboard already reads.
- **One `AdaptationLog`** per cascade run (not per strategy attempt) with:
  - `observation` = `{ "triggered_detectors": [...], "cascade_steps": ["s1_dynamic_k:NOT_IMPROVED", "s2_chunk_size:NOT_IMPROVED", "s3_combined:IMPROVED"] }`
  - `strategy_selected` = winner (or `"none"`)
  - `outcome` = `IMPROVED` / `DEGRADED` / `NO_CHANGE`
  - `rolled_back` = True if any intermediate S2/S3 was rolled back, regardless of final outcome.

The legacy multi-AdaptationLog-per-event pattern goes away — one row per
cascade pass is enough and keeps the dashboard's "recent adaptations" feed
readable.

---

## 9. Step-by-Step Execution Order (the actual coding sequence)

1. **DB model additions** — `StrategyCounter`, `RuntimeFlag` in `db/models.py`. Restart `uvicorn` to let `init_db()` create them. (No code path uses them yet — safe to ship alone.)
2. **`repair/cascade.py`** — write the orchestrator skeleton with stubs for each strategy (returning unchanged metrics so cascade always falls through to `unfixable`). Wire `RepairReport` + `AdaptationLog` writes. Unit-feasibility check: run against a known flagged event manually via an ad-hoc script.
3. **Strategy 1** — implement `run_s1_dynamic_k` using `_dynamic_k_selection` + `_probe_metrics`. No Pinecone writes. Test with a query you know is borderline.
4. **Strategy 2** — implement `run_s2_chunk_size`. Reuse `_get_chunk_ids_for_query`, `rechunk_semantic`, `reembed`. Confirm rollback path with a deliberately bad config (size=50).
5. **Strategy 3** — combine 1+2. Add explicit pre-rollback before calling `reembed` so S3 doesn't double-snapshot.
6. **Strategy 4** — `repair/llm_swap.py` + integration. Verify `get_fallback_llm()` actually loads `gemma3:27b` locally before relying on it (`ollama list`).
7. **Trigger logic in `auto_worker.py`** — switch `BATCH_THRESHOLD` to the count+ratio gate. Remove old per-event diagnose/select calls; replace with `run_repair_cascade(event.id)`.
8. **Counters + promotion** — `maybe_promote_dynamic_k()` in cascade; `_resolve_main_k` in `controllers/retrieval.py`. Test: artificially set `StrategyCounter.success_count = 5`, restart uvicorn, hit `/query`, confirm `k` is dynamic.
9. **Strip legacy `handle_event` cascade** — remove `auto_resolve`, query-reformulation, and llm-fallback blocks. Leave the function as a single "rechunk + probe + rollback" primitive used by S2/S3.
10. **Deprecate `select_strategy / check_cooldown / set_cooldown`** — add deprecation docstrings; delete call sites in `auto_worker.py`.
11. **Read-only endpoints** for `/strategy-counters` and `/runtime-flags` in `api/routes.py`. Useful for the dashboard and for debugging.
12. **Smoke test end-to-end** — see §10.

I'll mark each step with TaskCreate/TaskUpdate when I begin coding so progress
is visible per file.

---

## 10. Verification Plan (manual, no test suite exists)

Prereqs: `ollama serve` running with `mistral` and `gemma3:27b` pulled,
Pinecone credentials in `.env`, fresh `db/autorag.db` (or `python
db/clear_db.py`).

1. **Trigger gate**
   - Run `uvicorn main:app --reload` and `python auto_worker.py` in two terminals.
   - Send 10 queries: 7 healthy, 3 deliberately weird (likely to flag).
   - Confirm cascade does **not** fire (3/10 = 30% but < 5 minimum).
   - Send 3 more weird queries (6/13 = 46%, ≥5). Cascade should fire.
2. **S1 wins**
   - Pick a flagged complex query where K=5 misses context. S1 should resolve it; `StrategyCounter.s1_dynamic_k.success_count` increments by 1.
3. **S2 wins after S1 fails**
   - Send a short-factual query that flagged because chunks are too large. S1 won't help (K change doesn't shrink chunks); S2 should resolve via `reduce_chunk_size`.
4. **S3 wins**
   - Construct a cross-section query that needs both K bump and `large_coherent_chunks`. Verify S1 and S2 fail individually but S3 succeeds.
5. **S4 wins**
   - Find a query where chunks already contain the answer but `mistral` says "I don't know." S1/S2/S3 should fail; S4 (`gemma3:27b`) should produce a real answer. Confirm no Pinecone writes happened on S4.
6. **All fail → unfixable**
   - Ask a question completely outside the ingested corpus. All 4 strategies fail; event marked `unfixable=True`; `RepairReport.strategy_used = "none"`.
7. **Promotion**
   - Repeat Test #2 five times with different queries (or `UPDATE
     autorag_strategy_counters SET success_count = 5 WHERE strategy =
     's1_dynamic_k';`). Restart `uvicorn`. Confirm:
     - `/api/v1/query` now uses dynamic K (check logs / `/strategy-counters`).
     - Cascade for the next flagged event **skips** S1 and starts at S2.
8. **Rollback safety**
   - Force S2 to pick `chunk_size=50` via a debug override. S2 must roll back; index hash before vs. after must match. Subsequent S3 attempt must see the original chunks, not the bad S2 output.

---

## 11. What This Plan Deliberately Does NOT Do

- **No retry loop.** Each event gets exactly one cascade pass. If all four
  fail, the event is `unfixable` forever. The user explicitly asked for
  single-pass.
- **No demotion.** Once `dynamic_k_promoted` flips True, it stays True.
- **No promotion of S2/S3/S4** into the main pipeline. Only S1 is eligible
  per spec.
- **No dashboard changes.** The existing dashboard keeps working because
  `RepairReport` columns are unchanged.
- **No new metric**. Resolution = `_is_improved()` everywhere. No new
  threshold knobs in `config.py`.
- **No schema migrations** beyond two additive tables. Existing rows and
  columns are untouched (some are deprecated, not dropped).

---

## 12. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `_is_improved()` is too strict and every strategy "fails" → flood of `unfixable` | Medium | Keep the existing thresholds; if real-world false-negatives surface, relax in one place (single source of truth) |
| `gemma3:27b` not pulled locally → S4 always errors | High on a fresh install | `repair/llm_swap.py` catches the exception, returns unchanged metrics → S4 just "fails" cleanly; cascade ends in `unfixable`. Document the prereq in CLAUDE.md commands section. |
| Pinecone snapshot table grows without bound after many unfixable events | Low | Add a cleanup query in `auto_worker.py` that deletes `ChunkSnapshot` rows older than 7 days, or for events that are now `unfixable=True` and have nothing to roll back to |
| Counter race between `auto_worker.py` and a future parallel worker | Low | The single-worker model is unchanged. If multi-worker is added later, switch to `UPDATE ... WHERE` atomic counter SQL (already proposed) |
| Lifetime ratio dominates after one bad day, blocking all future repairs | Medium | The "open question" Q1 — if you say rolling window, switch denominator to last-N or last-T |

---

## 13. Quick Reference — Where Each Spec Bullet Lands

| Spec section | Lives in |
|---|---|
| §1 Initial baseline K=5 | `controllers/retrieval.py::_resolve_main_k` (returns 5 when flag unset) |
| §2 Trigger (≥5 AND ≥30%) | `auto_worker.py::run_batch_worker` |
| §2 Exclude `unfixable` from count | Same query, `unfixable == False` filter |
| §3 Strategy order | `repair/cascade.py::run_repair_cascade` |
| §3 Strategy 1 (dynamic K) | `repair/cascade.run_s1_dynamic_k` (reuses `_dynamic_k_selection`) |
| §3 Strategy 2 (chunk size, K=5) | `repair/cascade.run_s2_chunk_size` (reuses `diagnose` + `rechunk_semantic`) |
| §3 Strategy 3 (combined) | `repair/cascade.run_s3_combined` |
| §3 Strategy 4 (alt LLM) | `repair/cascade.run_s4_alt_llm` (`get_fallback_llm`) |
| §3 Counter increments | `repair/cascade.increment_counter` → `StrategyCounter` table |
| §3 Resolved check | `repair/orchestrator._is_improved` (reused) |
| §3 Single pass + mark unresolved | Cascade sets `unfixable=True` when all four fail |
| §3 Rollback to previous state | `repair/reembedder.rollback_from_snapshot` (already exists, called between failed S2/S3 attempts) |
| §4 Promotion threshold ≥5 | `repair/cascade.maybe_promote_dynamic_k` → `RuntimeFlag("dynamic_k_promoted")` |
| §4 Skip S1 after promotion | Cascade reads flag at the top of `run_repair_cascade` |
| §4 Main pipeline uses dynamic K | `controllers/retrieval._resolve_main_k` |
| §4 Only S1 promoted | Hard-coded — no other strategy ever sets a `*_promoted` flag |

---

**Awaiting answers on Q1–Q8 before I start editing code.** If you say "go
with the defaults," I'll start at Step §9.1 (DB model additions).
