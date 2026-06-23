"""
Repair Cascade — Single-Pass Ordered Strategy Execution
=========================================================
Replaces the legacy multi-retry loop with a single-pass cascade:

  S1: Dynamic K        — vary retrieval K by question category (no Pinecone change)
  S2: Chunk Size       — rechunk with decision-engine config, K fixed at 5
  S3: Combined         — dynamic K + rechunk together
  S4: Alt LLM          — same chunks, swap to gemma3:27b (no Pinecone change)

Each event gets exactly ONE cascade pass. The first strategy that
satisfies the resolution check wins, increments its counter, and resolves
the event. If all four fail → event is marked unfixable.

Resolution check:
  - S1/S2/S3: _is_improved(metrics_before, metrics_after) — composite check.
  - S4: bypasses _is_improved (chunks unchanged → top1_score is unchanged).
        Wins iff the swapped LLM produced a substantive (non-refusal) answer.

Rollback ownership:
  The cascade — not handle_event — owns rollback. Each S2/S3 attempt calls
  handle_event with internal_rollback=False, gets raw post-rechunk metrics,
  judges, and either commits (keeps chunks, breaks loop) or rolls back the
  snapshot before moving to the next strategy.

Promotion: When S1 accumulates ≥ 5 successes, dynamic K is promoted
to the main query pipeline (controllers/retrieval.py).
"""
import json
import time
import logging
from datetime import datetime

from db.session import get_session
from db.models import (
    LowRecallEvent, QueryLog, RepairReport, AdaptationLog,
    StrategyCounter, RuntimeFlag,
)
from repair.orchestrator import (
    _probe_metrics, _is_improved, _is_non_answer,
    _dynamic_k_selection, handle_event,
)
from repair.reembedder import rollback_from_snapshot
from detector.decision_engine import diagnose, STRATEGY_TO_RECHUNK
from services.llm_factory import get_vector_store, get_fallback_llm
from controllers.metrics import classify_question
from controllers.retrieval import _resolve_main_k
from config import settings



# ══════════════════════════════════════════════════════════════
#  COUNTER & PROMOTION HELPERS
# ══════════════════════════════════════════════════════════════

def increment_counter(strategy_name: str):
    """Increments the success counter for a strategy. Creates the row if missing."""
    session = get_session()
    try:
        counter = session.query(StrategyCounter).filter(
            StrategyCounter.strategy == strategy_name
        ).first()
        if counter:
            counter.success_count += 1
            counter.last_incremented_at = datetime.utcnow()
        else:
            counter = StrategyCounter(
                strategy=strategy_name,
                success_count=1,
                last_incremented_at=datetime.utcnow(),
            )
            session.add(counter)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.warning(f"[cascade] Failed to increment counter for {strategy_name}: {e}")
    finally:
        session.close()


def is_flag_set(flag_name: str) -> bool:
    """Checks if a RuntimeFlag is set to True."""
    session = get_session()
    try:
        flag = session.query(RuntimeFlag).filter(
            RuntimeFlag.name == flag_name,
            RuntimeFlag.value == True,
        ).first()
        return flag is not None
    finally:
        session.close()


def _set_flag(flag_name: str, value: bool):
    """Sets a RuntimeFlag. Creates the row if missing."""
    session = get_session()
    try:
        flag = session.query(RuntimeFlag).filter(
            RuntimeFlag.name == flag_name
        ).first()
        if flag:
            flag.value = value
            flag.set_at = datetime.utcnow()
        else:
            flag = RuntimeFlag(
                name=flag_name,
                value=value,
                set_at=datetime.utcnow(),
            )
            session.add(flag)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.warning(f"[cascade] Failed to set flag {flag_name}: {e}")
    finally:
        session.close()


def maybe_promote_dynamic_k():
    """Promotes Strategy 1 (dynamic K) to the main pipeline if threshold met."""
    session = get_session()
    try:
        promotion_threshold = settings.promotion_threshold
        counter = session.query(StrategyCounter).filter(
            StrategyCounter.strategy == "s1_dynamic_k"
        ).first()
        if (counter and counter.success_count >= promotion_threshold
                and not is_flag_set("dynamic_k_promoted")):
            _set_flag("dynamic_k_promoted", True)
            logging.info(
                f"[promotion] Strategy 1 (dynamic K) promoted to main pipeline! "
                f"({counter.success_count} successes)"
            )
            print(f"🎯 [PROMOTION] Dynamic K promoted to main query pipeline "
                  f"after {counter.success_count} successes!")
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════
#  INDIVIDUAL STRATEGY IMPLEMENTATIONS
#
#  Each strategy returns:
#    {
#      "metrics_before_local": dict,  # baseline at the SAME K as this strategy's
#                                     # probe — used by _is_improved so the
#                                     # comparison isn't biased by a K mismatch
#                                     # between cascade-level baseline and the
#                                     # strategy's probe. Falls back to cascade's
#                                     # metrics_before when the strategy doesn't
#                                     # change K (e.g., S4).
#      "metrics_after": dict,         # probe metrics after applying the strategy
#      "pinecone_touched": bool,      # True if Pinecone vectors were modified
#      "new_chunk_ids": list,         # IDs to roll back if strategy fails
#      "skip_improve_check": bool,    # True iff strategy wins via custom logic
#      "win": bool,                   # only consulted when skip_improve_check=True
#      "details": dict,               # k_used, chunk_size, chunk_overlap, ...
#    }
# ══════════════════════════════════════════════════════════════

def _run_s1_dynamic_k(event, log, diagnosis_result, metrics_before):
    """
    Strategy 1 — Dynamic K Selection.
    Varies retrieval K based on question category. No Pinecone modification.

    Comparison: cascade-level metrics_before (at the user-experienced K) vs.
    metrics_after at dynamic K. Different K on each side IS the experiment —
    the whole point of S1 is "did changing K help?", so K-bias is inherent
    and not a flaw to correct for.
    """
    q_category = (diagnosis_result.get("question_category")
                  or log.question_category
                  or classify_question(log.query))

    vs = get_vector_store()
    quick_results = vs.similarity_search_with_score(log.query, k=15)
    quick_scores = [round(float(s), 4) for _, s in quick_results]
    dynamic_k = _dynamic_k_selection(log.query, q_category, quick_scores)

    print(f"[cascade] S1: Trying dynamic K={dynamic_k} (category={q_category})")

    metrics_after = _probe_metrics(log.query, k=dynamic_k)

    return {
        "metrics_before_local": metrics_before,   # cascade baseline at user K
        "metrics_after": metrics_after,
        "pinecone_touched": False,
        "new_chunk_ids": [],
        "skip_improve_check": False,
        "win": False,
        "details": {
            "k_used": dynamic_k,
            "category": q_category,
            "chunk_size": None,
            "chunk_overlap": None,
            "chunks_before_count": 0,
            "chunks_after_count": 0,
        },
    }


def _run_s2_chunk_size(event, log, diagnosis_result, metrics_before):
    """
    Strategy 2 — Chunk Size Variation.
    Rechunks with decision-engine recommended config. K FIXED at 5.
    Cascade (not handle_event) owns rollback.
    """
    recommended = diagnosis_result.get("recommended_strategy", "reduce_chunk_size")
    config = diagnosis_result.get("recommended_config", {})
    rechunk_strategy = STRATEGY_TO_RECHUNK.get(recommended, "semantic")

    chunk_size = config.get("chunk_size", 1250)
    chunk_overlap = config.get("chunk_overlap", 200)

    print(f"[cascade] S2: Trying rechunk strategy={rechunk_strategy} "
          f"size={chunk_size} overlap={chunk_overlap} (K=5)")

    result = handle_event(
        event.id,
        strategy=rechunk_strategy,
        config={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        diagnosis=diagnosis_result,
        internal_rollback=False,   # cascade owns rollback
        k_override=5,              # S2 contract: K fixed at 5
    )

    if result.get("error"):
        logging.warning(f"[cascade] S2 error: {result['error']}")
        return {
            "metrics_before_local": metrics_before,
            "metrics_after": metrics_before,
            "pinecone_touched": False,
            "new_chunk_ids": [],
            "skip_improve_check": False,
            "win": False,
            "details": {"error": result["error"]},
        }

    # handle_event probed BEFORE at the same K=5 it probed AFTER — use that
    # as the local baseline so the comparison isolates the chunk-size effect.
    return {
        "metrics_before_local": result.get("metrics_before", metrics_before),
        "metrics_after": result.get("metrics_after", metrics_before),
        "pinecone_touched": result.get("pinecone_touched", False),
        "new_chunk_ids": result.get("new_chunk_ids", []),
        "skip_improve_check": False,
        "win": False,
        "details": {
            "k_used": result.get("k_used", 5),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "rechunk_strategy": rechunk_strategy,
            "chunks_before_count": result.get("chunks_before", 0),
            "chunks_after_count": result.get("chunks_after", 0),
        },
    }


def _run_s3_combined(event, log, diagnosis_result, metrics_before):
    """
    Strategy 3 — Combined: Dynamic K + Chunk Size.
    Both dynamic K selection and rechunking applied together.
    Cascade (not handle_event) owns rollback.
    """
    q_category = (diagnosis_result.get("question_category")
                  or log.question_category
                  or classify_question(log.query))

    vs = get_vector_store()
    quick_results = vs.similarity_search_with_score(log.query, k=15)
    quick_scores = [round(float(s), 4) for _, s in quick_results]
    dynamic_k = _dynamic_k_selection(log.query, q_category, quick_scores)

    recommended = diagnosis_result.get("recommended_strategy", "reduce_chunk_size")
    config = diagnosis_result.get("recommended_config", {})
    rechunk_strategy = STRATEGY_TO_RECHUNK.get(recommended, "semantic")

    chunk_size = config.get("chunk_size", 1250)
    chunk_overlap = config.get("chunk_overlap", 200)

    print(f"[cascade] S3: Trying combined — K={dynamic_k} + rechunk "
          f"size={chunk_size} overlap={chunk_overlap}")

    result = handle_event(
        event.id,
        strategy=rechunk_strategy,
        config={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        diagnosis=diagnosis_result,
        internal_rollback=False,   # cascade owns rollback
        k_override=dynamic_k,      # S3 contract: dynamic K + new chunks
    )

    if result.get("error"):
        logging.warning(f"[cascade] S3 error: {result['error']}")
        return {
            "metrics_before_local": metrics_before,
            "metrics_after": metrics_before,
            "pinecone_touched": False,
            "new_chunk_ids": [],
            "skip_improve_check": False,
            "win": False,
            "details": {"error": result["error"]},
        }

    # Both before and after probed at dynamic_k inside handle_event —
    # fair comparison isolates the combined chunk-size + K effect.
    return {
        "metrics_before_local": result.get("metrics_before", metrics_before),
        "metrics_after": result.get("metrics_after", metrics_before),
        "pinecone_touched": result.get("pinecone_touched", False),
        "new_chunk_ids": result.get("new_chunk_ids", []),
        "skip_improve_check": False,
        "win": False,
        "details": {
            "k_used": dynamic_k,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "rechunk_strategy": rechunk_strategy,
            "chunks_before_count": result.get("chunks_before", 0),
            "chunks_after_count": result.get("chunks_after", 0),
        },
    }


def _run_s4_alt_llm(event, log, diagnosis_result, metrics_before):
    """
    Strategy 4 — Alternate LLM.
    Same chunks, no Pinecone change. Swaps LLM for a larger model.

    S4 cannot use _is_improved as-is: top1_score is unchanged (same chunks),
    so the score-delta check always returns False. S4 wins iff the swapped
    LLM produces a substantive (non-refusal) answer.
    """
    chunks = metrics_before.get("chunks", [])
    if not chunks:
        return {
            "metrics_before_local": metrics_before,
            "metrics_after": metrics_before,
            "pinecone_touched": False,
            "new_chunk_ids": [],
            "skip_improve_check": True,
            "win": False,
            "details": {"error": "No chunks available for LLM swap"},
        }

    try:
        fallback_llm = get_fallback_llm()
        context_text = "\n\n".join(chunks)
        prompt = (
            "You are a precise factual assistant. Answer the question using ONLY "
            "the context below. Extract the answer directly from the text. Do NOT "
            "use any outside knowledge. If the exact answer appears in the context, "
            "state it clearly and concisely.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {log.query}\n\n"
            "Answer:"
        )

        model_name = settings.fallback_llm_model
        print(f"[cascade] S4: Trying alt LLM ({model_name}) on same chunks...")
        fallback_answer = fallback_llm.invoke(prompt).content.strip()

        metrics_after = dict(metrics_before)
        metrics_after["answer"] = fallback_answer

        # S4-specific win condition: substantive answer = success
        win = not _is_non_answer(fallback_answer)
        print(f"[cascade] S4 alt LLM answer: "
              f"{'SUBSTANTIVE' if win else 'NON-ANSWER'} "
              f"({len(fallback_answer)} chars)")

        return {
            "metrics_before_local": metrics_before,
            "metrics_after": metrics_after,
            "pinecone_touched": False,
            "new_chunk_ids": [],
            "skip_improve_check": True,
            "win": win,
            "details": {
                "llm_used": model_name,
                "k_used": 5,
                "chunk_size": None,
                "chunk_overlap": None,
                "chunks_before_count": len(chunks),
                "chunks_after_count": len(chunks),
            },
        }
    except Exception as e:
        logging.warning(f"[cascade] S4 alt LLM failed: {e}")
        return {
            "metrics_before_local": metrics_before,
            "metrics_after": metrics_before,
            "pinecone_touched": False,
            "new_chunk_ids": [],
            "skip_improve_check": True,
            "win": False,
            "details": {"error": str(e)},
        }


# ══════════════════════════════════════════════════════════════
#  MAIN CASCADE ENTRY POINT
# ══════════════════════════════════════════════════════════════

def run_repair_cascade(event_id: int) -> dict:
    """
    Single-pass ordered cascade for one LowRecallEvent.

    Tries S1→S2→S3→S4 in order. The first strategy that satisfies its
    resolution check wins, increments its counter, and resolves the event.
    All four failing → mark unfixable=True.

    Returns:
        {
            "event_id": int,
            "winning_strategy": str or None,
            "cascade_steps": list of step outcomes,
            "outcome": "IMPROVED" | "UNFIXABLE",
        }
    """
    session = get_session()
    t0 = time.time()

    try:
        event = session.query(LowRecallEvent).filter(
            LowRecallEvent.id == event_id
        ).first()
        if not event:
            return {"error": f"Event {event_id} not found"}
        if event.resolved:
            return {"message": f"Event {event_id} already resolved"}
        if event.unfixable:
            return {"message": f"Event {event_id} already marked unfixable"}

        log = session.query(QueryLog).filter(
            QueryLog.id == event.query_log_id
        ).first()
        if not log:
            return {"error": f"QueryLog for event {event_id} not found"}

        event.attempts = 1  # single pass

        # Baseline = what the user actually experienced. Pre-promotion this is
        # K=5; post-promotion (dynamic_k_promoted=True) it's the dynamic K
        # the main pipeline would have used. This matters for telemetry and
        # the RepairReport "before" snapshot, NOT for win-decisions —
        # each strategy probes its own K-matched baseline for that.
        baseline_k = _resolve_main_k(log.query)
        metrics_before = _probe_metrics(log.query, k=baseline_k)
        diagnosis_result = diagnose(event, log)

        # Build cascade, skipping S1 if dynamic K is already promoted to main pipeline
        skip_s1 = is_flag_set("dynamic_k_promoted")
        cascade = []
        if not skip_s1:
            cascade.append(("s1_dynamic_k", _run_s1_dynamic_k))
        cascade.extend([
            ("s2_chunk_size", _run_s2_chunk_size),
            ("s3_combined", _run_s3_combined),
            ("s4_alt_llm", _run_s4_alt_llm),
        ])

        winning_strategy = None
        winning_details = {}
        metrics_after = metrics_before
        cascade_steps = []
        any_rolled_back = False

        for name, fn in cascade:
            result = fn(event, log, diagnosis_result, metrics_before)
            step_metrics = result.get("metrics_after", metrics_before)
            local_before = result.get("metrics_before_local", metrics_before)

            # Resolution check: S4 has custom logic (skip_improve_check=True),
            # all others use _is_improved against the strategy's OWN K-matched
            # baseline so a K mismatch between cascade.metrics_before and the
            # strategy's probe doesn't bias the decision (S2 in particular
            # would lose recall mechanically when comparing K=5 vs dynamic).
            if result.get("skip_improve_check"):
                resolved = bool(result.get("win"))
            else:
                resolved = _is_improved(local_before, step_metrics)

            if resolved:
                winning_strategy = name
                winning_details = result.get("details", {})
                metrics_after = step_metrics
                cascade_steps.append(f"{name}:RESOLVED")
                print(f"[cascade] ✅ {name} RESOLVED event {event_id}")
                break

            cascade_steps.append(f"{name}:NOT_RESOLVED")
            print(f"[cascade] ❌ {name} did not resolve event {event_id}")

            # Roll back any Pinecone changes this failed strategy made.
            # This MUST happen before the next strategy runs so the snapshot
            # table doesn't accumulate stale rows and the next handle_event
            # call sees a clean baseline.
            if result.get("pinecone_touched"):
                new_ids = result.get("new_chunk_ids", [])
                if new_ids:
                    rollback_from_snapshot(event_id, new_chunk_ids=new_ids)
                    any_rolled_back = True
                    print(f"[cascade] 🔄 Rolled back {name} Pinecone changes")

        # Resolve or mark unfixable
        if winning_strategy:
            event.resolved = True
            increment_counter(winning_strategy)
            if winning_strategy == "s1_dynamic_k":
                maybe_promote_dynamic_k()
            outcome = "IMPROVED"
        else:
            event.unfixable = True
            outcome = "UNFIXABLE"

        # Provenance: RepairReport — populated from the winning strategy's details
        resolved_answer = metrics_after.get("answer") if winning_strategy else None
        report = RepairReport(
            event_id=event_id,
            strategy_used=winning_strategy or "none",
            chunks_before=int(winning_details.get("chunks_before_count", 0) or 0),
            chunks_after=int(winning_details.get("chunks_after_count", 0) or 0),
            score_before=round(metrics_before.get("top1_score", 0), 4),
            score_after=round(metrics_after.get("top1_score", 0), 4),
            resolved=winning_strategy is not None,
            original_answer=log.llm_response,
            resolved_answer=resolved_answer,
            precision_before=metrics_before.get("context_precision"),
            precision_after=metrics_after.get("context_precision"),
            recall_before=metrics_before.get("recall"),
            recall_after=metrics_after.get("recall"),
            accuracy_before=metrics_before.get("answer_accuracy"),
            accuracy_after=metrics_after.get("answer_accuracy"),
            dynamic_k=winning_details.get("k_used"),
            duration_ms=int((time.time() - t0) * 1000),
            chunks_before_text=json.dumps(metrics_before.get("chunks", [])),
            chunks_after_text=json.dumps(metrics_after.get("chunks", [])),
        )
        session.add(report)

        # Provenance: AdaptationLog — one per cascade pass
        adaptation = AdaptationLog(
            event_id=event_id,
            observation=json.dumps({
                "triggered_detectors": json.loads(event.triggered_detectors or "[]"),
                "score_before": round(metrics_before.get("top1_score", 0), 4),
                "cascade_steps": cascade_steps,
                "skipped_s1": skip_s1,
            }),
            diagnosis=json.dumps({
                "root_cause": diagnosis_result.get("root_cause", "unknown"),
                "question_category": diagnosis_result.get("question_category", "unknown"),
                "severity_score": diagnosis_result.get("severity_score", 0),
                "reasoning": diagnosis_result.get("reasoning", ""),
            }),
            strategy_selected=winning_strategy or "none",
            config_before=json.dumps({}),
            config_after=json.dumps(winning_details),
            metrics_before=json.dumps({"top1_score": round(metrics_before.get("top1_score", 0), 4)}),
            metrics_after=json.dumps({"top1_score": round(metrics_after.get("top1_score", 0), 4)}),
            outcome=outcome,
            rolled_back=any_rolled_back,
        )
        session.add(adaptation)
        session.commit()

        status_emoji = "✅" if winning_strategy else "❌"
        print(f"[cascade] {status_emoji} Event {event_id} | "
              f"Winner: {winning_strategy or 'NONE'} | "
              f"Steps: {cascade_steps} | "
              f"Duration: {int((time.time() - t0) * 1000)}ms")

        return {
            "event_id": event_id,
            "winning_strategy": winning_strategy,
            "cascade_steps": cascade_steps,
            "outcome": outcome,
            "score_before": round(metrics_before.get("top1_score", 0), 4),
            "score_after": round(metrics_after.get("top1_score", 0), 4),
            "duration_ms": int((time.time() - t0) * 1000),
        }

    except Exception as e:
        session.rollback()
        logging.error(f"[cascade] Error processing event {event_id}: {e}")
        return {"error": str(e)}
    finally:
        session.close()
