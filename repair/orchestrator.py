"""
Repair orchestrator — Month 1 (lightweight re-query approach).

Re-runs the original query to check if retrieval scores have improved
(e.g. after re-ingestion with a different chunking strategy).
Records before/after scores in a RepairReport.
"""

import json
import time
from db.session import get_session
from db.models import LowRecallEvent, QueryLog, RepairReport


def handle_event(event_id: int, strategy: str = "semantic") -> dict:
    """
    Handle a low-recall event.

    1. Look up the original query from the event.
    2. Re-run retrieval to get fresh scores.
    3. Compare with original scores.
    4. Create a RepairReport and mark resolved if improved.

    Args:
        event_id: ID of the LowRecallEvent to repair.
        strategy: Chunking strategy label (recorded for tracking).

    Returns:
        dict with repair results.
    """
    session = get_session()
    try:
        event = session.query(LowRecallEvent).filter(
            LowRecallEvent.id == event_id
        ).first()
        if not event:
            return {"error": f"Event {event_id} not found"}

        log = session.query(QueryLog).filter(
            QueryLog.id == event.query_log_id
        ).first()
        if not log:
            return {"error": f"Original query log not found for event {event_id}"}

        # Original scores
        original_scores = json.loads(log.top_k_scores or "[]")
        score_before = original_scores[0] if original_scores else 0.0
        chunks_before = len(original_scores)

        # Re-run the query (import here to avoid circular imports)
        from controllers.retrieval import answer_query

        start = time.perf_counter()
        result = answer_query(log.query)
        duration_ms = (time.perf_counter() - start) * 1000

        new_scores = result.get("scores", [])
        score_after = new_scores[0] if new_scores else 0.0
        chunks_after = len(new_scores)

        # Resolved if top score is now above the low-recall threshold
        resolved = score_after >= 0.45

        report = RepairReport(
            event_id=event_id,
            strategy_used=strategy,
            score_before=score_before,
            score_after=score_after,
            chunks_before=chunks_before,
            chunks_after=chunks_after,
            resolved=resolved,
            duration_ms=round(duration_ms, 2),
        )
        session.add(report)

        if resolved:
            event.resolved = True

        session.commit()

        return {
            "event_id": event_id,
            "strategy": strategy,
            "score_before": round(score_before, 4),
            "score_after": round(score_after, 4),
            "improvement": round(score_after - score_before, 4),
            "resolved": resolved,
            "duration_ms": round(duration_ms, 2),
        }
    except Exception as e:
        session.rollback()
        return {"error": str(e)}
    finally:
        session.close()
