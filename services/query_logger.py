"""
Query logger with low-recall detection rules.

Month 1 detection rules:
  1. low_top_score  — best chunk score < 0.45
  2. score_drop     — gap between rank-1 and rank-3 > 0.3
  3. llm_uncertainty — response contains hedging phrases
"""

import json
from db.session import get_session
from db.models import QueryLog, LowRecallEvent

# ── Detection thresholds ─────────────────────────────────────────────────
LOW_SCORE_THRESHOLD = 0.45
SCORE_DROP_THRESHOLD = 0.3
UNCERTAINTY_PHRASES = [
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "unclear", "cannot determine", "not enough information",
    "no information", "not mentioned", "i cannot", "i can't",
    "outside the provided context", "not specified",
]


def _detect_low_recall(scores: list[float], llm_response: str) -> list[str]:
    """Return list of triggered detector names."""
    triggered = []

    # Rule 1: low top score
    if scores and scores[0] < LOW_SCORE_THRESHOLD:
        triggered.append("low_top_score")

    # Rule 2: score drop between rank-1 and rank-3
    if len(scores) >= 3 and (scores[0] - scores[2]) > SCORE_DROP_THRESHOLD:
        triggered.append("score_drop")

    # Rule 3: LLM uncertainty language
    response_lower = (llm_response or "").lower()
    if any(phrase in response_lower for phrase in UNCERTAINTY_PHRASES):
        triggered.append("llm_uncertainty")

    # Bonus rule: very short answer
    if llm_response and len(llm_response.strip()) < 50:
        triggered.append("short_answer")

    return triggered


def _compute_severity(triggered: list[str], top_score: float | None) -> str:
    """HIGH if 2+ triggers, MEDIUM if 1 trigger + low score, else LOW."""
    if len(triggered) >= 2:
        return "HIGH"
    if len(triggered) == 1 and (top_score is not None and top_score < LOW_SCORE_THRESHOLD):
        return "MEDIUM"
    if triggered:
        return "LOW"
    return "LOW"


def log_query(
    *,
    query: str,
    llm_response: str,
    top_k_scores: list[float],
    retrieved_chunks: list[str],
    latency_ms: float,
    ctx_q_sim: float | None = None,
    answer_sem_sim: float | None = None,
) -> QueryLog:
    """
    Persist a query to the database and run low-recall detection.
    Returns the created QueryLog row.
    """
    triggered = _detect_low_recall(top_k_scores, llm_response)
    flagged = len(triggered) > 0

    session = get_session()
    try:
        log = QueryLog(
            query=query,
            llm_response=llm_response,
            top_k_scores=json.dumps(top_k_scores),
            retrieved_chunks=json.dumps(retrieved_chunks),
            flagged=flagged,
            latency_ms=round(latency_ms, 2),
            ctx_q_sim=ctx_q_sim,
            answer_sem_sim=answer_sem_sim,
        )
        session.add(log)
        session.flush()  # get log.id

        if flagged:
            top_score = top_k_scores[0] if top_k_scores else None
            severity = _compute_severity(triggered, top_score)
            event = LowRecallEvent(
                query_log_id=log.id,
                triggered_detectors=json.dumps(triggered),
                severity=severity,
            )
            session.add(event)

        session.commit()
        return log
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
