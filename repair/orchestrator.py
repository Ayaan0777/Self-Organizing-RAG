"""
Repair Orchestrator — SAFE repair loop with ROLLBACK & PROVENANCE
===================================================================
When a low-recall event is detected, this module:
  1. Loads the failing query from QueryLog
  2. Retrieves the poorly-matching chunks (with their Pinecone IDs)
  3. Rechunks the text with a DYNAMIC strategy and config from the decision engine
  4. Replaces ONLY those specific chunks (with snapshot for rollback)
  5. Probes recall to check if the repair improved results
  6. ROLLS BACK if metrics degraded (safe experimentation guarantee)
  7. Writes full provenance (AdaptationLog + RepairReport)

SAFETY: Only the specific failing chunks are replaced. The rest of
the index stays intact. Failed repairs are automatically reverted.
"""
import json
import time

from db.session import get_session
from db.models import LowRecallEvent, QueryLog, RepairReport, AdaptationLog
from repair.chunker import rechunk_semantic, rechunk_llm, rechunk_entropy
from repair.reembedder import reembed, rollback_from_snapshot
from controllers.retrieval import generate_answer_only
from services.llm_factory import get_vector_store, get_pinecone_index, get_embeddings
from config import settings

STRATEGY_MAP = {
    "semantic": rechunk_semantic,
    "llm":      rechunk_llm,
    "entropy":  rechunk_entropy,
}


def _probe_score(query: str, namespace: str = None) -> float:
    """Re-runs the original failing query and returns the new top-1 score."""
    vs     = get_vector_store(namespace)
    result = vs.similarity_search_with_score(query, k=1)
    # Pinecone returns cosine similarity (1 = identical)
    return round(float(result[0][1]), 4) if result else 0.0


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
      3. Rechunk with DYNAMIC config from the decision engine
      4. Snapshot + replace chunks (safe — rollback-enabled)
      5. Probe recall to check improvement
      6. ROLLBACK if metrics degraded
      7. Write RepairReport + AdaptationLog (full provenance)

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

        score_before = json.loads(log.top_k_scores or "[0]")[0]

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

        # Probe: did the repair actually improve recall?
        score_after = _probe_score(log.query)
        improved    = score_after > score_before + 0.05
        rolled_back = False
        resolved_answer = None

        # ROLLBACK if repair degraded metrics
        if not improved:
            print(f"[repair] Repair did NOT improve score ({score_before:.3f} → {score_after:.3f}). "
                  f"Rolling back...")
            rollback_result = rollback_from_snapshot(event_id)
            rolled_back = True
            score_after = _probe_score(log.query)  # re-probe after rollback
            print(f"[repair] Rollback complete. Score after rollback: {score_after:.3f}")
        else:
            # Generate the improved answer
            repaired_result = generate_answer_only(log.query)
            resolved_answer = repaired_result.get("answer")

        # Determine outcome for provenance
        if improved:
            outcome = "IMPROVED"
        elif rolled_back:
            outcome = "DEGRADED"
        else:
            outcome = "NO_CHANGE"

        # Persist repair report (backward compatible)
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
            duration_ms   = int((time.time() - t0) * 1000),
        )
        event.resolved = improved
        session.add(report)

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
            metrics_before=json.dumps({"top1_score": round(score_before, 4)}),
            metrics_after=json.dumps({"top1_score": round(score_after, 4)}),
            outcome=outcome,
            rolled_back=rolled_back,
        )
        session.add(adaptation)
        session.commit()

        status = "RESOLVED" if improved else ("ROLLED_BACK" if rolled_back else "UNRESOLVED")
        print(f"[repair] event={event_id} strategy={strategy} "
              f"size={chunk_size} overlap={chunk_overlap} "
              f"score {score_before:.3f} -> {score_after:.3f} {status}")

        return {
            "event_id"     : event_id,
            "strategy"     : strategy,
            "chunk_size"   : chunk_size,
            "chunk_overlap": chunk_overlap,
            "score_before" : round(score_before, 4),
            "score_after"  : round(score_after, 4),
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

