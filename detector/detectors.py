"""
Low Recall Detector
========================================
5 detection rules, each independent and non-blocking:

  1. low_top_score      — Top-1 retrieval score below threshold
  2. score_drop         — Largest adjacent-rank score gap
  3. llm_uncertainty    — LLM response contains hedging language
  4. semantic_mismatch  — Retrieved chunks are semantically fragmented
  5. evidence_mismatch  — LLM answer doesn't match retrieved evidence
"""
import json
import numpy as np
from datetime import datetime, timedelta

from db.session import get_session
from db.models import QueryLog, LowRecallEvent
from config import settings

# ── Detection thresholds — calibrated for mxbai-embed-large ──
# NOTE: These thresholds are now read from the config settings object.
# Default values:
#   score_low = 0.65 (rule 1: top-1 score below this → flag)
#   score_drop = 0.15 (rule 2: gap rank-1 to rank-K above this → flag)
#   coherence_ratio = 0.65 (rule 4: mean_pairwise_sim / top1_score must be ≥ this)
#   evidence_match = 0.60 (rule 5: answer↔evidence sim below this → flag)

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


def _detect_semantic_mismatch(chunks: list[str], top1_score: float) -> bool:
    """
    Rule 4 — Semantic Mismatch Detector (relative threshold)
    Flags when the lower-ranked chunks are inconsistent with the top match.
    Computes mean pairwise chunk similarity and compares it against
    coherence_ratio × top1_score.

    Why relative instead of absolute: cross-section queries naturally pull
    chunks from different sub-topics within one document (e.g. stadium /
    game / players paragraphs of one Wikipedia article). Their mean pairwise
    sim sits ~0.55–0.65, tripping any absolute threshold above ~0.55. By
    tying the bar to top1 we expect coherence proportional to retrieval
    quality: high top1 → expect tight chunks; modest top1 → tolerate spread.

    Skips when top1 ≤ 0 (no retrieval) — Rule 1 owns that case.
    """
    if len(chunks) < 2 or top1_score <= 0:
        return False
    try:
        emb_model = _get_embeddings_model()
        # Single batched embed call — one Ollama roundtrip instead of N.
        raw = emb_model.embed_documents([c[:500] for c in chunks])
        embeddings = [np.array(e) for e in raw]

        sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sims.append(_cosine_sim(embeddings[i], embeddings[j]))

        mean_sim = float(np.mean(sims)) if sims else 1.0
        threshold = settings.coherence_ratio * top1_score
        return mean_sim < threshold
    except Exception:
        return False


def _detect_evidence_mismatch(answer: str, chunks: list[str]) -> bool:
    """
    Rule 5 — LLM Response–Evidence Mismatch Detector
    Checks whether the answer is grounded in AT LEAST ONE retrieved chunk.

    Compares the answer embedding against each chunk individually and takes
    the maximum similarity. If even the best-matching chunk falls below the
    threshold, the answer isn't backed by anything we retrieved → flag.

    Why per-chunk and not concatenated-evidence: a short factual answer
    (e.g. "$5,000,000") embeds far from a long concatenated context blob
    purely because of length asymmetry — the old "embed all chunks together"
    version flagged correct answers reflexively. "Grounded in some chunk"
    is what we actually mean by "evidence backs the answer."
    """
    if not answer or not chunks:
        return False
    try:
        emb_model = _get_embeddings_model()
        # Single batched embed call: answer + all chunks in one roundtrip.
        texts = [answer[:500]] + [c[:500] for c in chunks]
        raw = emb_model.embed_documents(texts)
        answer_emb = np.array(raw[0])
        chunk_embs = [np.array(e) for e in raw[1:]]
        max_sim = max(
            (_cosine_sim(answer_emb, ce) for ce in chunk_embs),
            default=0.0,
        )
        return max_sim < settings.evidence_match
    except Exception:
        return False


def run_detectors(log_id: int):
    """
    Runs all 5 detection rules against a freshly logged query.
    Writes a LowRecallEvent if any rules trigger. Marks the QueryLog row as flagged.
    Called automatically at the end of answer_query() in controllers/retrieval.py.
    Silent on failure — never raises, never blocks the API response.
    """
    if log_id <= 0:
        return  # upstream logging failed (sentinel -1) or invalid id

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
        if scores and scores[0] < settings.score_low:
            triggered.append("low_top_score")

        # Rule 2 — Big drop between adjacent ranks (retrieval cliffs).
        # Uses max adjacent gap, not rank-1 minus rank-K. The latter scales
        # with K, so once dynamic K is promoted the rule's sensitivity drifts
        # (K=2 → smaller spread, K=10 → wider). Max adjacent gap is K-invariant.
        if len(scores) >= 2:
            max_gap = max(scores[i] - scores[i + 1] for i in range(len(scores) - 1))
            if max_gap > settings.score_drop:
                triggered.append("score_drop")

        # Rule 3 — LLM response contains uncertainty / hedging language
        if any(phrase in response for phrase in UNCERTAINTY_PHRASES):
            triggered.append("llm_uncertainty")

        # Rule 4 — Retrieved chunks are semantically inconsistent with top1
        if chunks and scores and _detect_semantic_mismatch(chunks, scores[0]):
            triggered.append("semantic_mismatch")

        # Rule 5 — LLM answer doesn't match the retrieved evidence
        if chunks and _detect_evidence_mismatch(log.llm_response or "", chunks):
            triggered.append("evidence_mismatch")



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
