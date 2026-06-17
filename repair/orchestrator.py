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

# ── Dynamic K: maps question category to optimal K bounds ──
CATEGORY_K_BOUNDS = {
    "short_factual": (2, 4),    # precise, focused retrieval
    "complex":       (5, 8),    # need more context
    "cross_section": (6, 10),   # spanning multiple topics
}
DEFAULT_K_BOUNDS = (3, 6)

# Score thresholds for dynamic K pruning
SCORE_NOISE_FLOOR = 0.25   # chunks below this are noise
SCORE_CLIFF_THRESHOLD = 0.12  # gap that signals a quality cliff


def _dynamic_k_selection(
    query: str,
    question_category: str = None,
    scores: list = None,
) -> int:
    """
    Dynamic K Selection — decides how many chunks to retrieve based on
    query complexity and (optionally) score distribution.

    Stage 1: Set K bounds from question category
    Stage 2: If scores are provided, prune below noise floor
    Stage 3: Detect score cliff (largest drop > threshold)

    Returns:
        Optimal K (integer)
    """
    # Stage 1: Category-based K bounds
    k_min, k_max = CATEGORY_K_BOUNDS.get(
        question_category or "", DEFAULT_K_BOUNDS
    )
    target_k = (k_min + k_max) // 2  # midpoint as default

    # If no scores to analyze, return the category midpoint
    if not scores:
        return target_k

    # Stage 2: Absolute score pruning (remove noise)
    valid = [s for s in scores if s >= SCORE_NOISE_FLOOR]
    if len(valid) < k_min:
        return max(k_min, len(valid)) if valid else k_min

    # Stage 3: Score cliff detection
    best_cliff_k = len(valid)  # default: keep all valid chunks
    for i in range(1, len(valid)):
        gap = valid[i - 1] - valid[i]
        if gap > SCORE_CLIFF_THRESHOLD:
            best_cliff_k = i  # cut AFTER the cliff
            break

    # Clamp to bounds
    k = max(k_min, min(best_cliff_k, k_max))
    return k


# ══════════════════════════════════════════════════════════════
#  ENHANCED MULTI-METRIC PROBE
# ══════════════════════════════════════════════════════════════

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def _probe_metrics(query: str, ground_truths: list = None,
                   namespace: str = None, k: int = 5) -> dict:
    """
    Enhanced probe that evaluates the current retrieval quality using
    multiple metrics instead of just the top-1 score.
    Uses dynamic K to retrieve the optimal number of chunks.

    Returns:
        {
            "top1_score": float,       # raw retrieval similarity
            "context_precision": float, # fraction of top-K chunks that are relevant
            "recall": float,           # fraction of ground truths covered by chunks
            "answer_accuracy": float,  # semantic similarity of answer vs ground truth
            "answer": str,             # the generated answer (for provenance)
            "chunks": list[str],       # retrieved chunk texts (dynamic K)
            "scores": list[float],     # per-chunk similarity scores
            "k_used": int,             # the K value actually used
        }
    """
    vs = get_vector_store(namespace)
    # Over-fetch to allow score cliff detection, then prune
    fetch_k = min(k * 2, 15)
    results = vs.similarity_search_with_score(query, k=fetch_k)

    if not results:
        return {
            "top1_score": 0.0, "context_precision": 0.0,
            "recall": 0.0, "answer_accuracy": 0.0,
            "answer": "", "chunks": [], "scores": [], "k_used": k,
        }

    # Prune to the requested K
    results = results[:k]
    top1_score = round(float(results[0][1]), 4)
    chunks = [doc.page_content for doc, _ in results]
    scores = [round(float(s), 4) for _, s in results]

    # If ground truths are available, compute full metrics
    if ground_truths and len(ground_truths) > 0:
        # Context Precision: are the top-K chunks relevant?
        ctx_precision = retrieval_precision_at_k(chunks, ground_truths)

        # Recall: are all ground truths covered by at least one chunk?
        recall = retrieval_recall_at_k(chunks, ground_truths)

        # Answer Accuracy: generate answer and compare to ground truth.
        # Pass k explicitly so the answer is generated from the SAME chunks
        # we measured precision/recall against (without this, the answer
        # would use _resolve_main_k's K — different from probe K post-promotion).
        try:
            rag_result = generate_answer_only(query, namespace, k=k)
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
            "scores": scores,
            "k_used": k,
        }
    else:
        # Still generate the answer even without ground truths — needed
        # for storing the resolved_answer in the repair report. Pass k so
        # the answer comes from the same K as the chunks we just measured.
        try:
            rag_result = generate_answer_only(query, namespace, k=k)
            answer = rag_result.get("answer", "")
        except Exception as e:
            logging.warning(f"[probe] answer generation failed (no GT path): {e}")
            answer = ""

        return {
            "top1_score": top1_score,
            "context_precision": None,
            "recall": None,
            "answer_accuracy": None,
            "answer": answer,
            "chunks": chunks,
            "scores": scores,
            "k_used": k,
        }


# ── Phrases that indicate the LLM couldn't find an answer ──
# Short refusals (< 150 chars containing these = definite non-answer)
_NON_ANSWER_PHRASES = [
    "i don't know",
    "i do not know",
    "i cannot answer",
    "i'm unable to",
    "i am unable to",
    "cannot find the answer",
    "no information available",
    "insufficient information to answer",
    "unable to determine the answer",
    "cannot determine the answer",
    "does not provide information",
    "does not contain information",
    "does not mention",
    "doesn't provide information",
    "doesn't contain information",
    "no specific information",
    "not mentioned in the",
    "not provided in the",
]


def _is_non_answer(answer: str) -> bool:
    """
    Detects if an LLM answer is a non-answer.
    If the chunk contains the answer, the LLM answers directly — no hedging.
    Any hedging phrase = the answer isn't in the chunks = non-answer.
    """
    if not answer or len(answer.strip()) < 10:
        return True
    lower = answer.lower().strip()
    return any(phrase in lower for phrase in _NON_ANSWER_PHRASES)


def _is_improved(before: dict, after: dict) -> bool:
    """
    Determines if the repair improved metrics using a composite check.

    Checks THREE things:
      1. Did the retrieval score improve?
      2. Did precision/recall/accuracy improve (if ground truth available)?
      3. Is the new answer actually a real answer (not "I don't know")?

    A repair is REJECTED if the score improved but the LLM still
    can't produce a real answer from the retrieved chunks.
    """
    # CRITICAL: Even if scores improved, if the answer is still a non-answer
    # then the repair didn't actually help — reject it.
    after_answer = after.get("answer", "")
    if _is_non_answer(after_answer):
        return False

    has_gt = before.get("context_precision") is not None

    if not has_gt:
        # Any positive improvement counts
        score_went_up = after["top1_score"] > before["top1_score"] + 0.001
        score_now_good = after["top1_score"] >= 0.7 and before["top1_score"] < 0.7
        return score_went_up or score_now_good

    # Full metric comparison — relaxed thresholds
    prec_improved = (after["context_precision"] or 0) > (before["context_precision"] or 0) + 0.01
    recall_improved = (after["recall"] or 0) > (before["recall"] or 0) + 0.01
    acc_improved = (after["answer_accuracy"] or 0) > (before["answer_accuracy"] or 0) + 0.01
    score_improved = after["top1_score"] > before["top1_score"] + 0.001

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
    Returns ground-truth answers for a query, or [] if none are available.

    Looks up the dataset (controllers/gt_lookup.py loads dataset/long_ans.json
    on first call) by normalized question text. When a match is found, the
    cascade's _probe_metrics gets real precision/recall/accuracy signals
    against the reference answers instead of falling back to the top1-only
    no-GT path. This makes _is_improved much sharper on dataset-matched
    queries.

    Falls back to [] silently on any failure — no GT path still works fine.
    """
    try:
        from controllers.gt_lookup import lookup_ground_truth
        gts = lookup_ground_truth(query_log.query)
        return gts if gts else []
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════
#  MAIN REPAIR HANDLER
# ══════════════════════════════════════════════════════════════

def handle_event(
    event_id: int,
    strategy: str = "semantic",
    config: dict = None,
    diagnosis: dict = None,
    internal_rollback: bool = True,
    k_override: int = None,
) -> dict:
    """
    Rechunk + reembed + probe primitive for one LowRecallEvent.

      1. Load event + failing query
      2. Pick K (k_override if given, else dynamic K from question category)
      3. Probe BEFORE metrics at chosen K
      4. Rechunk with given chunk config + strategy
      5. Replace chunks in Pinecone with snapshot for rollback
      6. Probe AFTER metrics at chosen K
      7. If internal_rollback=True (legacy): judge with _is_improved and roll
         back on failure.
         If internal_rollback=False (cascade-owned): return raw post-rechunk
         metrics; the caller decides whether to keep or roll back.

    Args:
        event_id: ID of the LowRecallEvent to repair.
        strategy: Rechunking strategy name ("semantic", "llm", "entropy").
        config: {"chunk_size": int, "chunk_overlap": int, "chunk_strategy": str}.
        diagnosis: Diagnosis dict from decision engine (for question_category).
        internal_rollback: When True (default), self-judges and self-restores.
            When False, returns raw post-rechunk metrics; caller owns rollback.
            The cascade in repair/cascade.py uses False so it can compare each
            strategy on equal terms before deciding to commit or revert.
        k_override: When set, use this K instead of the category-based dynamic K.
            Strategy 2 in the cascade pins this to 5.
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

        # ── PICK K: k_override (S2) or category-based dynamic K ──
        q_category = diagnosis.get("question_category") or log.question_category or "unknown"
        if k_override is not None:
            k_used = k_override
            logging.info(
                f"[repair] Event {event_id} | K override={k_used} (category={q_category})"
            )
        else:
            vs_quick = get_vector_store()
            quick_results = vs_quick.similarity_search_with_score(log.query, k=15)
            quick_scores = [round(float(s), 4) for _, s in quick_results]
            k_used = _dynamic_k_selection(log.query, q_category, quick_scores)
            logging.info(
                f"[repair] Event {event_id} | Dynamic K={k_used} (category={q_category})"
            )

        # ── PROBE BEFORE: Measure current metrics at chosen K ──
        metrics_before = _probe_metrics(log.query, ground_truths, k=k_used)
        score_before = metrics_before["top1_score"]
        chunks_before_text = metrics_before.get("chunks", [])

        logging.info(
            f"[repair] Event {event_id} | BEFORE metrics: "
            f"top1={metrics_before['top1_score']:.3f} "
            f"precision={metrics_before.get('context_precision', 'N/A')} "
            f"recall={metrics_before.get('recall', 'N/A')} "
            f"accuracy={metrics_before.get('answer_accuracy', 'N/A')}"
        )

        # Get the SPECIFIC chunk IDs and text for the failing query
        chunk_ids, full_text, source = _get_chunk_ids_for_query(
            log.query, k=k_used
        )

        if not full_text.strip():
            return {"error": "Could not retrieve chunk text for repair — "
                             "chunks may not have text metadata stored"}

        if not chunk_ids:
            return {"error": "No chunks found for this query — ingest documents first"}

        # Extract dynamic chunk config (with backward-compatible defaults)
        chunk_size = config.get("chunk_size", 1250)
        chunk_overlap = config.get("chunk_overlap", 200)

        print(f"[repair] Found {len(chunk_ids)} chunks to replace for event {event_id} "
              f"(size={chunk_size}, overlap={chunk_overlap})")

        # Rechunk the text with the chosen strategy + dynamic config
        rechunk_fn = STRATEGY_MAP.get(strategy, rechunk_semantic)
        new_chunks = rechunk_fn(full_text, source,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap)

        # Replace ONLY the specific chunks WITH SNAPSHOT for rollback
        counts = reembed(new_chunks, source,
                         old_chunk_ids=chunk_ids,
                         event_id=event_id)

        # ── PROBE AFTER: Measure raw post-rechunk metrics at same K ──
        metrics_after = _probe_metrics(log.query, ground_truths, k=k_used)
        score_after = metrics_after["top1_score"]
        chunks_after_text = metrics_after.get("chunks", [])

        # ── DECISION + ROLLBACK (only if cascade hasn't claimed ownership) ──
        improved = _is_improved(metrics_before, metrics_after)
        rolled_back = False
        new_ids = counts.get("new_chunk_ids", [])

        logging.info(
            f"[repair] Event {event_id} | AFTER metrics: "
            f"top1={metrics_after['top1_score']:.3f} "
            f"precision={metrics_after.get('context_precision', 'N/A')} "
            f"recall={metrics_after.get('recall', 'N/A')} "
            f"accuracy={metrics_after.get('answer_accuracy', 'N/A')} "
            f"| improved={improved}"
        )

        if internal_rollback and not improved:
            print(f"[repair] Rechunk did NOT improve metrics "
                  f"(score {score_before:.3f} -> {score_after:.3f}). Rolling back...")
            rollback_from_snapshot(event_id, new_chunk_ids=new_ids)
            rolled_back = True
            metrics_final = _probe_metrics(log.query, ground_truths, k=k_used)
            score_after = metrics_final["top1_score"]
            print(f"[repair] Rollback complete. Score after rollback: {score_after:.3f}")

        resolved_answer = metrics_after.get("answer") if improved else None

        if improved:
            outcome = "IMPROVED"
        elif rolled_back:
            outcome = "DEGRADED"
        else:
            outcome = "NO_CHANGE"

        status = "RESOLVED" if improved else ("ROLLED_BACK" if rolled_back else "UNRESOLVED")
        print(f"[repair] event={event_id} strategy={strategy} "
              f"size={chunk_size} overlap={chunk_overlap} K={k_used} "
              f"score {score_before:.3f} -> {score_after:.3f} "
              f"{status}")

        return {
            "event_id"     : event_id,
            "strategy"     : strategy,
            "chunk_size"   : chunk_size,
            "chunk_overlap": chunk_overlap,
            "dynamic_k"    : k_used,
            "k_used"       : k_used,
            "score_before" : round(score_before, 4),
            "score_after"  : round(score_after, 4),
            "improved"     : improved,
            "rolled_back"  : rolled_back,
            "outcome"      : outcome,
            "chunks_before": counts["old_count"],
            "chunks_after" : counts["new_count"],
            "chunks_before_text": chunks_before_text,
            "chunks_after_text" : chunks_after_text,
            "metrics_before": metrics_before,
            "metrics_after" : metrics_after,
            "resolved_answer": resolved_answer,
            "new_chunk_ids": new_ids,
            "pinecone_touched": True,
        }

    except Exception as e:
        session.rollback()
        return {"error": str(e)}
    finally:
        session.close()

