"""
Low Recall Detector — Month 2 Prototype
========================================
6 detection rules, each independent and non-blocking:

  Existing (Month 1):
    1. low_top_score      — Top-1 retrieval score below threshold
    2. score_drop         — Large gap between rank-1 and rank-K
    3. llm_uncertainty    — LLM response contains hedging language

  New (Month 2):
    4. semantic_mismatch  — Retrieved chunks are semantically fragmented
    5. evidence_mismatch  — LLM answer doesn't match retrieved evidence
    6. user_frustration   — User reformulated a similar query recently
"""
import json
import numpy as np
from datetime import datetime, timedelta

from db.session import get_session
from db.models import QueryLog, LowRecallEvent

# ── Detection thresholds — tune after observing real query scores ──
SCORE_LOW          = 0.45   # rule 1: top-1 score below this → flag
SCORE_DROP         = 0.3    # rule 2: gap rank-1 to rank-K above this → flag
CHUNK_COHERENCE    = 0.55   # rule 4: mean pairwise chunk sim below this → flag
EVIDENCE_MATCH     = 0.50   # rule 5: answer↔evidence sim below this → flag
FRUSTRATION_SIM    = 0.85   # rule 6: cosine sim threshold for "same query"
FRUSTRATION_WINDOW = 300    # rule 6: seconds to look back for reformulations

UNCERTAINTY_PHRASES = [
    # Direct uncertainty
    "i don't know", "i'm not sure", "cannot find", "no information",
    "not available", "i cannot", "unclear", "no relevant", "don't have",
    "unable to find", "no context", "not enough information",
    # Refusal / hedging patterns
    "does not provide", "does not contain", "does not mention",
    "does not specify", "does not include", "does not state",
    "do not provide", "do not contain", "do not mention",
    "doesn't provide", "doesn't contain", "doesn't mention",
    "doesn't specify", "doesn't include",
    "not mentioned", "not specified", "not provided", "not stated",
    "no specific", "no mention of", "no evidence",
    "cannot determine", "cannot be determined", "not explicitly",
    "the text does not", "the context does not", "the passage does not",
    "based on the provided", "not found in",
    "there is no", "there are no",
    "i could not", "could not find",
]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def _get_embeddings_model():
    """Lazy import to avoid circular deps at module load time."""
    from services.llm_factory import get_embeddings
    return get_embeddings()


def _detect_semantic_mismatch(chunks: list[str]) -> bool:
    """
    Rule 4 — Semantic Mismatch Detector
    Checks whether the top-K retrieved chunks are semantically coherent.
    If chunks are about wildly different topics (low mean pairwise similarity),
    retrieval is fragmented and the LLM gets confused context.
    """
    if len(chunks) < 2:
        return False
    try:
        emb_model = _get_embeddings_model()
        embeddings = [np.array(emb_model.embed_query(c[:500])) for c in chunks]

        sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sims.append(_cosine_sim(embeddings[i], embeddings[j]))

        mean_sim = np.mean(sims) if sims else 1.0
        return float(mean_sim) < CHUNK_COHERENCE
    except Exception:
        return False


def _detect_evidence_mismatch(answer: str, chunks: list[str]) -> bool:
    """
    Rule 5 — LLM Response–Evidence Mismatch Detector
    Compares the LLM's answer embedding against the concatenated evidence.
    If the answer is semantically far from the evidence, the LLM may be
    hallucinating or answering from parametric knowledge instead of context.
    """
    if not answer or not chunks:
        return False
    try:
        emb_model = _get_embeddings_model()
        answer_emb = np.array(emb_model.embed_query(answer[:500]))
        evidence_text = " ".join(chunks)[:1000]
        evidence_emb = np.array(emb_model.embed_query(evidence_text))

        sim = _cosine_sim(answer_emb, evidence_emb)
        return sim < EVIDENCE_MATCH
    except Exception:
        return False


def _detect_user_frustration(query: str, session) -> bool:
    """
    Rule 6 — User Frustration Signal Detector
    Checks if a semantically similar query was asked within the last N seconds.
    Repeated/reformulated queries indicate the user is unsatisfied with results.
    """
    try:
        cutoff = datetime.utcnow() - timedelta(seconds=FRUSTRATION_WINDOW)
        recent = session.query(QueryLog).filter(
            QueryLog.timestamp >= cutoff
        ).order_by(QueryLog.timestamp.desc()).limit(20).all()

        if len(recent) < 2:
            return False

        emb_model = _get_embeddings_model()
        query_emb = np.array(emb_model.embed_query(query[:500]))

        # Compare against recent queries (skip the current one, which is the most recent)
        for row in recent[1:]:
            prev_emb = np.array(emb_model.embed_query(row.query[:500]))
            sim = _cosine_sim(query_emb, prev_emb)
            if sim >= FRUSTRATION_SIM:
                return True

        return False
    except Exception:
        return False


def run_detectors(log_id: int):
    """
    Runs all 6 detection rules against a freshly logged query.
    Writes a LowRecallEvent if any rules trigger. Marks the QueryLog row as flagged.
    Called automatically at the end of answer_query() in controllers/retrieval.py.
    Silent on failure — never raises, never blocks the API response.
    """
    if log_id < 0:
        return  # upstream logging failed, nothing to detect on

    session = get_session()
    try:
        log = session.query(QueryLog).filter(QueryLog.id == log_id).first()
        if not log:
            return

        scores   = json.loads(log.top_k_scores or "[]")
        response = (log.llm_response or "").lower().strip()
        chunks   = json.loads(log.retrieved_chunks or "[]") if log.retrieved_chunks else []
        triggered = []

        # Rule 1 — Top retrieval score is below acceptable threshold
        if scores and scores[0] < SCORE_LOW:
            triggered.append("low_top_score")

        # Rule 2 — Big drop between rank-1 and rank-K (retrieval is inconsistent)
        if len(scores) >= 2 and (scores[0] - scores[-1]) > SCORE_DROP:
            triggered.append("score_drop")

        # Rule 3 — LLM response contains uncertainty / hedging language
        if any(phrase in response for phrase in UNCERTAINTY_PHRASES):
            triggered.append("llm_uncertainty")

        # Rule 4 — Retrieved chunks are semantically fragmented
        if chunks and _detect_semantic_mismatch(chunks):
            triggered.append("semantic_mismatch")

        # Rule 5 — LLM answer doesn't match the retrieved evidence
        if chunks and _detect_evidence_mismatch(log.llm_response or "", chunks):
            triggered.append("evidence_mismatch")

        # Rule 6 — User seems to be re-asking the same thing (frustration)
        if _detect_user_frustration(log.query, session):
            triggered.append("user_frustration")

        if not triggered:
            return  # healthy query, nothing to do

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
        print(f"[detector] ⚠ event={event.id} severity={severity} triggers={triggered}")

    except Exception as e:
        session.rollback()
        print(f"[detector] non-fatal warning: {e}")
    finally:
        session.close()
