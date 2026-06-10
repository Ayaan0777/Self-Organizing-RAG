"""
Low Recall Detector — SH2: Mentor's 3 Metrics
===============================================
Detection rules based on mentor's MEASURE stage:

  1. Retrieval Precision — proportion of top-k chunks that are relevant
  2. Context Sufficiency — LLM judges if context can fully answer the query
  3. Hallucination Rate  — LLM checks if answer is grounded in context

These detector names feed directly into the DECIDE stage (repair/orchestrator.py)
to select the appropriate repair strategy:
  - "low_retrieval_precision" → uses query-complexity-based chunk sizing
  - "context_insufficient"   → increases chunk size (more context)
  - "hallucination_detected" → decreases chunk size (less noise)
"""
import json
import logging

from db.session import get_session
from db.models import QueryLog, LowRecallEvent

# ── Detection thresholds ───────────────────────────────────────────
RELEVANCE_SCORE_FLOOR = 0.45    # chunks below this are considered irrelevant
MIN_PRECISION_RATIO   = 0.50    # at least 50% of top-k chunks must be relevant
TOP_SCORE_FLOOR       = 0.55    # if best chunk score is below this, retrieval failed

# ── LLM uncertainty phrases ────────────────────────────────────────
# If the LLM's own answer contains these, the context was clearly insufficient.
_UNCERTAINTY_PHRASES = [
    "does not mention", "doesn't mention",
    "no information", "not found in",
    "context does not", "context doesn't",
    "not mentioned", "no relevant",
    "cannot answer", "unable to answer",
    "not provided", "not available in",
    "no data", "outside the scope",
]


def _get_llm():
    """Lazy import to avoid circular deps at module load time."""
    from services.llm_factory import get_llm
    return get_llm()


def _check_retrieval_precision(scores: list[float]) -> bool:
    """
    Metric 1 — Retrieval Precision
    Two checks:
      a) Proportion of top-k chunks with score >= 0.45 must be >= 50%
      b) The BEST chunk score must be >= 0.55 (absolute floor)

    If the best score is only ~0.49 (like for irrelevant queries),
    that alone means the retrieval completely failed.

    Returns True if precision is LOW (should flag).
    """
    if not scores:
        return True  # no scores = definitely bad retrieval

    # Check a: absolute top-score floor
    if max(scores) < TOP_SCORE_FLOOR:
        return True  # best chunk is below 0.55 — retrieval failed

    # Check b: proportion of relevant chunks
    relevant_count = sum(1 for s in scores if s >= RELEVANCE_SCORE_FLOOR)
    precision = relevant_count / len(scores)
    return precision < MIN_PRECISION_RATIO


def _check_llm_uncertainty(answer: str) -> bool:
    """
    Quick keyword check — if the LLM's own answer admits the context
    is missing or irrelevant, that's a clear signal of bad retrieval.
    No extra LLM call needed.

    Returns True if uncertainty is DETECTED (should flag).
    """
    if not answer:
        return False
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in _UNCERTAINTY_PHRASES)


def _check_context_sufficiency(query: str, chunks: list[str]) -> bool:
    """
    Metric 2 — Context Sufficiency (Lenient)
    Uses the LLM to judge whether the retrieved context contains enough
    information to produce a reasonable answer to the question.

    NOTE: We only flag when the context is clearly MISSING key information.
    For broad questions like "explain about X", if the context contains
    relevant facts about X, that's sufficient — it doesn't need to cover
    every possible aspect.

    Returns True if context is INSUFFICIENT (should flag).
    """
    if not chunks:
        return True  # no context = definitely insufficient

    try:
        llm = _get_llm()
        context_text = "\n---\n".join(c[:500] for c in chunks[:5])

        prompt = (
            f"Question: {query}\n\n"
            f"Retrieved Context:\n{context_text}\n\n"
            "Does the context above contain RELEVANT information to answer "
            "this question? The context doesn't need to cover every aspect — "
            "just enough key facts to produce a useful answer.\n"
            "Reply 'NO' ONLY if the context is completely irrelevant or "
            "missing critical information needed to answer the question at all.\n"
            "Reply 'YES' if the context has at least some relevant facts.\n"
            "Reply with ONLY 'YES' or 'NO'."
        )
        result = llm.invoke(prompt).content.strip().upper()
        return "NO" in result
    except Exception as e:
        logging.warning(f"[detector] context sufficiency check failed: {e}")
        return False  # fail-safe: don't flag on LLM errors


def _check_hallucination(answer: str, chunks: list[str]) -> bool:
    """
    Metric 3 — Hallucination Rate (Lenient)
    Uses the LLM to check whether the generated answer CONTRADICTS or
    contains WRONG information compared to the retrieved context.

    NOTE: We intentionally allow the LLM to add supplementary information
    from its knowledge (e.g., naming an institution) as long as it doesn't
    contradict the context. Only flag outright fabrication or wrong facts.

    Returns True if hallucination is DETECTED (should flag).
    """
    if not answer or not chunks:
        return False  # nothing to check

    try:
        llm = _get_llm()
        context_text = "\n---\n".join(c[:500] for c in chunks[:5])

        prompt = (
            f"Retrieved Context:\n{context_text}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            "Does the answer contain any information that CONTRADICTS or is "
            "FACTUALLY WRONG compared to the context above? "
            "Ignore any additional details the answer adds that don't conflict "
            "with the context — only flag if the answer says something that the "
            "context explicitly disagrees with or if the answer fabricates specific "
            "facts (dates, numbers, names) that are wrong.\n"
            "Reply with ONLY 'YES' if there are contradictions/wrong facts, "
            "or 'NO' if the answer is consistent with the context."
        )
        result = llm.invoke(prompt).content.strip().upper()
        return "YES" in result
    except Exception as e:
        logging.warning(f"[detector] hallucination check failed: {e}")
        return False  # fail-safe


def run_detectors(log_id: int):
    """
    Runs mentor's 3 detection metrics against a freshly logged query.
    Writes a LowRecallEvent if any metric triggers. Marks QueryLog as flagged.
    Called automatically at the end of answer_query() in controllers/retrieval.py.

    Detector tags written to LowRecallEvent.triggered_detectors:
      - "low_retrieval_precision" → DECIDE uses query-complexity chunk sizing
      - "context_insufficient"   → DECIDE increases chunk size
      - "hallucination_detected" → DECIDE decreases chunk size
    """
    if log_id < 0:
        return  # upstream logging failed

    session = get_session()
    try:
        log = session.query(QueryLog).filter(QueryLog.id == log_id).first()
        if not log:
            return

        scores   = json.loads(log.top_k_scores or "[]")
        answer   = (log.llm_response or "").strip()
        chunks   = json.loads(log.retrieved_chunks or "[]") if log.retrieved_chunks else []
        triggered = []

        # Metric 1 — Retrieval Precision (score ratio + absolute floor)
        if _check_retrieval_precision(scores):
            triggered.append("low_retrieval_precision")

        # Metric 2 — Context Sufficiency (LLM-based)
        if _check_context_sufficiency(log.query, chunks):
            triggered.append("context_insufficient")

        # Metric 3 — Hallucination Rate (LLM-based)
        if _check_hallucination(answer, chunks):
            triggered.append("hallucination_detected")

        # Bonus — LLM self-admitted uncertainty (fast keyword check)
        if _check_llm_uncertainty(answer):
            if "context_insufficient" not in triggered:
                triggered.append("context_insufficient")

        if not triggered:
            return  # healthy query

        severity = {1: "LOW", 2: "MEDIUM"}.get(len(triggered), "HIGH")

        event = LowRecallEvent(
            query_log_id        = log.id,
            triggered_detectors = json.dumps(triggered),
            severity            = severity,
            resolved            = False,
        )
        log.flagged = True
        session.add(event)
        session.commit()
        logging.info(f"[detector] ⚠ event={event.id} severity={severity} triggers={triggered}")

    except Exception as e:
        session.rollback()
        logging.warning(f"[detector] non-fatal warning: {e}")
    finally:
        session.close()
