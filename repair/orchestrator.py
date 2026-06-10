"""
Repair Orchestrator — SAFE repair loop with ROLLBACK & PROVENANCE
===================================================================
When a low-recall event is detected, this module:
  1. Loads the failing query from QueryLog
  2. Retrieves the poorly-matching chunks (with their Pinecone IDs)
  3. Rechunks the text with a DYNAMIC strategy and config from the decision engine
  4. Replaces ONLY those specific chunks (with snapshot for rollback)
  5. Probes recall, precision, accuracy to check if the repair improved results
  6. ROLLS BACK if metrics degraded (safe experimentation guarantee)
  7. Writes full provenance (AdaptationLog + RepairReport)

SAFETY: Only the specific failing chunks are replaced. The rest of
the index stays intact. Failed repairs are automatically reverted.

ENHANCED METRICS (v2):
  - Context Precision: fraction of top-K chunks relevant to the answer
  - Recall: fraction of ground-truths covered by at least one chunk
  - Answer Accuracy: semantic similarity of generated answer vs ground truth
  - Top-1 Score: raw retrieval similarity (backward compat)
"""
import json
import time
import numpy as np
import logging

from db.session import get_session
from db.models import LowRecallEvent, QueryLog, RepairReport, AdaptationLog
from repair.chunker import rechunk_semantic, rechunk_llm, rechunk_entropy
from repair.reembedder import reembed, rollback_from_snapshot
from controllers.retrieval import generate_answer_only
from services.llm_factory import get_vector_store, get_pinecone_index, get_embeddings
from controllers.metrics import (
    retrieval_precision_at_k,
    retrieval_recall_at_k,
)
from config import settings

STRATEGY_MAP = {
    "semantic": rechunk_semantic,
    "llm":      rechunk_llm,
    "entropy":  rechunk_entropy,
}


# ══════════════════════════════════════════════════════════════
#  ENHANCED MULTI-METRIC PROBE
# ══════════════════════════════════════════════════════════════

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def _probe_metrics(query: str, ground_truths: list = None, namespace: str = None) -> dict:
    """
    Enhanced probe that evaluates the current retrieval quality using
    multiple metrics instead of just the top-1 score.

    Returns:
        {
            "top1_score": float,       # raw retrieval similarity
            "context_precision": float, # fraction of top-K chunks that are relevant
            "recall": float,           # fraction of ground truths covered by chunks
            "answer_accuracy": float,  # semantic similarity of answer vs ground truth
            "answer": str,             # the generated answer (for provenance)
            "chunks": list[str],       # retrieved chunk texts
        }
    """
    vs = get_vector_store(namespace)
    results = vs.similarity_search_with_score(query, k=5)

    if not results:
        return {
            "top1_score": 0.0, "context_precision": 0.0,
            "recall": 0.0, "answer_accuracy": 0.0,
            "answer": "", "chunks": [],
        }

    top1_score = round(float(results[0][1]), 4)
    chunks = [doc.page_content for doc, _ in results]

    # If ground truths are available, compute full metrics
    if ground_truths and len(ground_truths) > 0:
        # Context Precision: are the top-K chunks relevant?
        ctx_precision = retrieval_precision_at_k(chunks, ground_truths)

        # Recall: are all ground truths covered by at least one chunk?
        recall = retrieval_recall_at_k(chunks, ground_truths)

        # Answer Accuracy: generate answer and compare to ground truth
        try:
            rag_result = generate_answer_only(query, namespace)
            answer = rag_result.get("answer", "")

            emb_model = get_embeddings()
            answer_emb = np.array(emb_model.embed_query(answer[:500]))
            # Compare against best ground truth
            best_accuracy = 0.0
            for gt in ground_truths:
                gt_emb = np.array(emb_model.embed_query(gt[:500]))
                sim = _cosine_sim(answer_emb, gt_emb)
                best_accuracy = max(best_accuracy, sim)

            answer_accuracy = round(best_accuracy, 4)
        except Exception as e:
            logging.warning(f"[probe] answer accuracy computation failed: {e}")
            answer = ""
            answer_accuracy = 0.0

        return {
            "top1_score": top1_score,
            "context_precision": ctx_precision,
            "recall": recall,
            "answer_accuracy": answer_accuracy,
            "answer": answer,
            "chunks": chunks,
        }
    else:
        # No ground truth — fallback to top-1 score only
        return {
            "top1_score": top1_score,
            "context_precision": None,
            "recall": None,
            "answer_accuracy": None,
            "answer": "",
            "chunks": chunks,
        }


def _is_improved(before: dict, after: dict) -> bool:
    """
    Determines if the repair improved metrics using a composite check.

    When ground truth is available:
      - Improved if ANY of precision, recall, or accuracy improved meaningfully
      - AND none of them degraded significantly

    When no ground truth:
      - Fallback to top-1 score improvement (> 0.05)
    """
    has_gt = before.get("context_precision") is not None

    if not has_gt:
        # Fallback: classic top-1 score check
        return after["top1_score"] > before["top1_score"] + 0.05

    # Full metric comparison
    prec_improved = (after["context_precision"] or 0) > (before["context_precision"] or 0) + 0.05
    recall_improved = (after["recall"] or 0) > (before["recall"] or 0) + 0.05
    acc_improved = (after["answer_accuracy"] or 0) > (before["answer_accuracy"] or 0) + 0.05
    score_improved = after["top1_score"] > before["top1_score"] + 0.05

    # Check for significant degradation in any metric
    prec_degraded = (after["context_precision"] or 0) < (before["context_precision"] or 0) - 0.1
    recall_degraded = (after["recall"] or 0) < (before["recall"] or 0) - 0.1
    acc_degraded = (after["answer_accuracy"] or 0) < (before["answer_accuracy"] or 0) - 0.1

    # Improved if at least one metric got better AND nothing degraded badly
    any_improved = prec_improved or recall_improved or acc_improved or score_improved
    any_degraded = prec_degraded or recall_degraded or acc_degraded

    return any_improved and not any_degraded


# ══════════════════════════════════════════════════════════════
#  CHUNK ID RETRIEVAL
# ══════════════════════════════════════════════════════════════

def _get_chunk_ids_for_query(query: str, namespace: str = None, k: int = 5) -> tuple:
    """
    Retrieves chunks matching the query and returns their Pinecone vector IDs
    along with the concatenated text for rechunking.

    Returns: (chunk_ids: list[str], full_text: str, source: str)
    """
    ns = namespace or settings.pinecone_namespace
    index = get_pinecone_index()
    embeddings = get_embeddings()

    # Embed the query and search Pinecone directly to get vector IDs
    query_emb = embeddings.embed_query(query)
    results = index.query(
        vector=query_emb,
        top_k=k,
        namespace=ns,
        include_metadata=True,
    )

    if not results.matches:
        return [], "", "unknown"

    chunk_ids = [m.id for m in results.matches]
    # Reconstruct the text from metadata
    texts = []
    source = "unknown"
    for m in results.matches:
        text = m.metadata.get("text", "")
        if not text:
            # LangChain stores text in the 'text' metadata field
            text = m.metadata.get("page_content", "")
        texts.append(text)
        if m.metadata.get("source"):
            source = m.metadata["source"]

    full_text = " ".join(texts)
    return chunk_ids, full_text, source


# ══════════════════════════════════════════════════════════════
#  GROUND TRUTH LOOKUP
# ══════════════════════════════════════════════════════════════

def _get_ground_truths_for_query(query_log) -> list:
    """
    Attempts to retrieve ground-truth answers for a query.
    Uses the answer_sem_sim presence as a signal that eval was run with GT.
    Falls back to empty list if no ground truth is available.
    """
    # If we have a stored ground-truth semantic similarity, the eval was run
    # with ground truth — try to find it from the original answer
    # For now, use the LLM response as a proxy reference when answer_sem_sim exists
    if query_log.answer_sem_sim is not None and query_log.llm_response:
        return [query_log.llm_response]
    return []


# ══════════════════════════════════════════════════════════════
#  MAIN REPAIR HANDLER
# ══════════════════════════════════════════════════════════════

def handle_event(
    event_id: int,
    strategy: str = "semantic",
    config: dict = None,
    diagnosis: dict = None,
) -> dict:
    """
    SAFE repair loop with rollback and provenance for one LowRecallEvent:
      1. Load the event and its original failing query
      2. Find the specific chunk IDs that were retrieved (poorly)
      3. Probe BEFORE metrics (precision, recall, accuracy, top-1)
      4. Rechunk with DYNAMIC config from the decision engine
      5. Snapshot + replace chunks (safe — rollback-enabled)
      6. Probe AFTER metrics
      7. ROLLBACK if metrics degraded (composite check)
      8. Write RepairReport + AdaptationLog (full provenance)

    Args:
        event_id: ID of the LowRecallEvent to repair.
        strategy: Rechunking strategy name ("semantic", "llm", "entropy").
        config: Dynamic chunk config from decision engine:
                {"chunk_size": int, "chunk_overlap": int, "chunk_strategy": str}
        diagnosis: Diagnosis dict from decision engine (for provenance logging).

    SAFETY: Only the specific failing chunks are replaced.
    If repair degrades metrics, it is automatically rolled back.
    """
    session = get_session()
    t0      = time.time()
    config  = config or {}
    diagnosis = diagnosis or {}

    try:
        event = session.query(LowRecallEvent).filter(
                    LowRecallEvent.id == event_id).first()
        if not event:
            return {"error": f"Event {event_id} not found"}
        if event.resolved:
            return {"message": f"Event {event_id} is already resolved — skipping"}

        log = session.query(QueryLog).filter(
                  QueryLog.id == event.query_log_id).first()
        if not log:
            return {"error": "Original query log entry missing"}

        # Get ground truths for enhanced metric evaluation
        ground_truths = _get_ground_truths_for_query(log)
        has_gt = len(ground_truths) > 0

        # ── PROBE BEFORE: Measure current metrics ──
        metrics_before = _probe_metrics(log.query, ground_truths)
        score_before = metrics_before["top1_score"]

        logging.info(
            f"[repair] Event {event_id} | BEFORE metrics: "
            f"top1={metrics_before['top1_score']:.3f} "
            f"precision={metrics_before.get('context_precision', 'N/A')} "
            f"recall={metrics_before.get('recall', 'N/A')} "
            f"accuracy={metrics_before.get('answer_accuracy', 'N/A')}"
        )

        # Get the SPECIFIC chunk IDs and text for the failing query
        chunk_ids, full_text, source = _get_chunk_ids_for_query(log.query)

        if not full_text.strip():
            return {"error": "Could not retrieve chunk text for repair — "
                             "chunks may not have text metadata stored"}

        if not chunk_ids:
            return {"error": "No chunks found for this query — ingest documents first"}

        # Extract dynamic chunk config (with backward-compatible defaults)
        chunk_size = config.get("chunk_size", 250)
        chunk_overlap = config.get("chunk_overlap", 80)

        print(f"[repair] Found {len(chunk_ids)} chunks to replace for event {event_id} "
              f"(size={chunk_size}, overlap={chunk_overlap})")

        # Save current config state for provenance
        config_before = {
            "chunk_size": 250,  # previous default
            "chunk_overlap": 80,
            "strategy": strategy,
        }
        config_after = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "strategy": strategy,
        }

        # Rechunk the text with the chosen strategy + dynamic config
        rechunk_fn = STRATEGY_MAP.get(strategy, rechunk_semantic)
        new_chunks = rechunk_fn(full_text, source,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap)

        # Replace ONLY the specific chunks WITH SNAPSHOT for rollback
        counts = reembed(new_chunks, source,
                         old_chunk_ids=chunk_ids,
                         event_id=event_id)

        # ── PROBE AFTER: Measure metrics post-repair ──
        metrics_after = _probe_metrics(log.query, ground_truths)
        score_after = metrics_after["top1_score"]

        # ── DECISION: Improved or rollback? ──
        improved = _is_improved(metrics_before, metrics_after)
        rolled_back = False
        resolved_answer = None

        logging.info(
            f"[repair] Event {event_id} | AFTER metrics: "
            f"top1={metrics_after['top1_score']:.3f} "
            f"precision={metrics_after.get('context_precision', 'N/A')} "
            f"recall={metrics_after.get('recall', 'N/A')} "
            f"accuracy={metrics_after.get('answer_accuracy', 'N/A')} "
            f"| improved={improved}"
        )

        # ROLLBACK if repair degraded metrics
        if not improved:
            print(f"[repair] Repair did NOT improve metrics "
                  f"(score {score_before:.3f} → {score_after:.3f}). Rolling back...")
            rollback_result = rollback_from_snapshot(event_id)
            rolled_back = True
            # Re-probe after rollback to get accurate final score
            metrics_final = _probe_metrics(log.query, ground_truths)
            score_after = metrics_final["top1_score"]
            print(f"[repair] Rollback complete. Score after rollback: {score_after:.3f}")
        else:
            # Success — keep the repaired answer
            resolved_answer = metrics_after.get("answer")

        # Determine outcome for provenance
        if improved:
            outcome = "IMPROVED"
        elif rolled_back:
            outcome = "DEGRADED"
        else:
            outcome = "NO_CHANGE"

        # Persist repair report with enhanced metrics
        report = RepairReport(
            event_id      = event_id,
            strategy_used = strategy,
            chunks_before = counts["old_count"],
            chunks_after  = counts["new_count"],
            score_before  = round(score_before, 4),
            score_after   = round(score_after, 4),
            resolved      = improved,
            original_answer = log.llm_response,
            resolved_answer = resolved_answer,
            # Enhanced metrics
            precision_before = metrics_before.get("context_precision"),
            precision_after  = metrics_after.get("context_precision"),
            recall_before    = metrics_before.get("recall"),
            recall_after     = metrics_after.get("recall"),
            accuracy_before  = metrics_before.get("answer_accuracy"),
            accuracy_after   = metrics_after.get("answer_accuracy"),
            duration_ms   = int((time.time() - t0) * 1000),
        )
        event.resolved = improved
        session.add(report)

        # Build full metrics dicts for provenance logging
        metrics_before_log = {"top1_score": round(score_before, 4)}
        metrics_after_log = {"top1_score": round(score_after, 4)}
        if has_gt:
            metrics_before_log.update({
                "context_precision": metrics_before.get("context_precision"),
                "recall": metrics_before.get("recall"),
                "answer_accuracy": metrics_before.get("answer_accuracy"),
            })
            metrics_after_log.update({
                "context_precision": metrics_after.get("context_precision"),
                "recall": metrics_after.get("recall"),
                "answer_accuracy": metrics_after.get("answer_accuracy"),
            })

        # PROVENANCE: Write AdaptationLog with full audit trail
        adaptation = AdaptationLog(
            event_id=event_id,
            observation=json.dumps({
                "triggered_detectors": json.loads(event.triggered_detectors or "[]"),
                "score_before": round(score_before, 4),
                "retrieval_precision": log.retrieval_precision,
                "context_sufficiency": log.context_sufficiency,
                "hallucination_rate": log.hallucination_rate,
                "question_category": log.question_category,
            }),
            diagnosis=json.dumps({
                "root_cause": diagnosis.get("root_cause", "unknown"),
                "question_category": diagnosis.get("question_category", "unknown"),
                "severity_score": diagnosis.get("severity_score", 0),
                "reasoning": diagnosis.get("reasoning", ""),
            }),
            strategy_selected=strategy,
            config_before=json.dumps(config_before),
            config_after=json.dumps(config_after),
            metrics_before=json.dumps(metrics_before_log),
            metrics_after=json.dumps(metrics_after_log),
            outcome=outcome,
            rolled_back=rolled_back,
        )
        session.add(adaptation)
        session.commit()

        status = "RESOLVED" if improved else ("ROLLED_BACK" if rolled_back else "UNRESOLVED")
        print(f"[repair] event={event_id} strategy={strategy} "
              f"size={chunk_size} overlap={chunk_overlap} "
              f"score {score_before:.3f} -> {score_after:.3f} "
              f"prec={metrics_before.get('context_precision', 'N/A')}->"
              f"{metrics_after.get('context_precision', 'N/A')} "
              f"recall={metrics_before.get('recall', 'N/A')}->"
              f"{metrics_after.get('recall', 'N/A')} "
              f"acc={metrics_before.get('answer_accuracy', 'N/A')}->"
              f"{metrics_after.get('answer_accuracy', 'N/A')} "
              f"{status}")

        return {
            "event_id"     : event_id,
            "strategy"     : strategy,
            "chunk_size"   : chunk_size,
            "chunk_overlap": chunk_overlap,
            "score_before" : round(score_before, 4),
            "score_after"  : round(score_after, 4),
            "precision_before": metrics_before.get("context_precision"),
            "precision_after": metrics_after.get("context_precision"),
            "recall_before": metrics_before.get("recall"),
            "recall_after": metrics_after.get("recall"),
            "accuracy_before": metrics_before.get("answer_accuracy"),
            "accuracy_after": metrics_after.get("answer_accuracy"),
            "improved"     : improved,
            "rolled_back"  : rolled_back,
            "outcome"      : outcome,
            "chunks_before": counts["old_count"],
            "chunks_after" : counts["new_count"],
            "duration_ms"  : report.duration_ms,
        }

    except Exception as e:
        session.rollback()
        return {"error": str(e)}
    finally:
        session.close()
