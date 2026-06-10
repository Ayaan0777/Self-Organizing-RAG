"""
Stage 2 Metrics — MEASURE Feedback Loop
========================================
Three new quality metrics + question classifier:

  1. retrieval_precision_at_k  — fraction of top-K chunks that are relevant
  2. context_sufficiency       — can the context fully answer the question?
  3. hallucination_rate        — fraction of answer claims not grounded in context
  4. classify_question         — short_factual | complex | cross_section

All metrics are embedding-based for speed (no extra LLM calls).
"""
import re
import numpy as np
from services.llm_factory import get_embeddings


# ── Thresholds (tunable via config.py) ──
PRECISION_RELEVANCE_THRESHOLD = 0.50   # chunk↔ground_truth sim above this = relevant
SUFFICIENCY_THRESHOLD         = 0.70   # context↔ground_truth sim above this = sufficient
HALLUCINATION_GROUNDING_THRESHOLD = 0.55  # claim↔context sim above this = grounded


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def _get_embedding(text: str) -> np.ndarray:
    """Get embedding vector for a text string."""
    emb_model = get_embeddings()
    return np.array(emb_model.embed_query(text[:500]))


# ══════════════════════════════════════════════════════════════
#  1. RETRIEVAL PRECISION @ K
# ══════════════════════════════════════════════════════════════

def retrieval_precision_at_k(
    chunks: list[str],
    ground_truths: list[str],
    threshold: float = PRECISION_RELEVANCE_THRESHOLD,
) -> float:
    """
    Measures what fraction of retrieved chunks are relevant to the answer.

    For each chunk, we compute its semantic similarity to every ground-truth
    answer. A chunk is "relevant" if its best match ≥ threshold.

    Args:
        chunks: List of retrieved chunk text strings (top-K).
        ground_truths: List of reference answer strings.
        threshold: Minimum similarity to count as relevant.

    Returns:
        Precision score (0.0–1.0). E.g., 0.6 means 3 out of 5 chunks are relevant.
    """
    if not chunks or not ground_truths:
        return 0.0

    try:
        gt_embeddings = [_get_embedding(gt) for gt in ground_truths]
        relevant_count = 0

        for chunk in chunks:
            chunk_emb = _get_embedding(chunk)
            # Check if this chunk is similar to ANY ground truth
            best_sim = max(
                _cosine_sim(chunk_emb, gt_emb) for gt_emb in gt_embeddings
            )
            if best_sim >= threshold:
                relevant_count += 1

        return round(relevant_count / len(chunks), 4)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════
#  2. CONTEXT SUFFICIENCY
# ══════════════════════════════════════════════════════════════

def context_sufficiency(
    chunks: list[str],
    ground_truths: list[str],
    threshold: float = SUFFICIENCY_THRESHOLD,
) -> bool:
    """
    Checks whether the retrieved context contains enough information
    to fully answer the question.

    Concatenates all chunks and compares against each ground-truth answer.
    Context is sufficient if the combined context embedding is semantically
    close enough to at least one ground truth.

    Args:
        chunks: List of retrieved chunk text strings.
        ground_truths: List of reference answer strings.
        threshold: Minimum similarity for context to be deemed sufficient.

    Returns:
        True if context is sufficient, False otherwise.
    """
    if not chunks or not ground_truths:
        return False

    try:
        # Combine all chunks into a single context representation
        combined_context = " ".join(chunks)[:2000]
        context_emb = _get_embedding(combined_context)

        # Check against each ground truth — sufficient if ANY matches well
        for gt in ground_truths:
            gt_emb = _get_embedding(gt)
            sim = _cosine_sim(context_emb, gt_emb)
            if sim >= threshold:
                return True

        return False
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════
#  3. HALLUCINATION RATE
# ══════════════════════════════════════════════════════════════

def _split_into_claims(answer: str) -> list[str]:
    """
    Splits an answer into individual claims (sentences).
    Each sentence is treated as one claim to verify.
    """
    if not answer or not answer.strip():
        return []

    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    # Filter out very short fragments (< 10 chars) that aren't real claims
    return [s.strip() for s in sentences if len(s.strip()) >= 10]


def hallucination_rate(
    answer: str,
    chunks: list[str],
    threshold: float = HALLUCINATION_GROUNDING_THRESHOLD,
) -> float:
    """
    Measures the fraction of claims in the answer that are NOT grounded
    in the retrieved context.

    Process:
      1. Split the answer into individual claims (sentences)
      2. For each claim, compute similarity to each retrieved chunk
      3. A claim is "grounded" if its best chunk similarity ≥ threshold
      4. Hallucination rate = ungrounded_claims / total_claims

    Args:
        answer: The generated LLM answer.
        chunks: List of retrieved chunk text strings.
        threshold: Minimum similarity for a claim to be considered grounded.

    Returns:
        Hallucination rate (0.0–1.0). E.g., 0.2 means 20% of claims are ungrounded.
        Returns 0.0 if there are no claims to check.
    """
    if not answer or not chunks:
        return 0.0

    claims = _split_into_claims(answer)
    if not claims:
        return 0.0

    try:
        chunk_embeddings = [_get_embedding(chunk) for chunk in chunks]
        ungrounded = 0

        for claim in claims:
            claim_emb = _get_embedding(claim)
            # Check if this claim is supported by ANY chunk
            best_sim = max(
                _cosine_sim(claim_emb, chunk_emb) for chunk_emb in chunk_embeddings
            )
            if best_sim < threshold:
                ungrounded += 1

        return round(ungrounded / len(claims), 4)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════
#  4. QUESTION CLASSIFIER
# ══════════════════════════════════════════════════════════════

# Keywords for classification heuristics
_FACTUAL_STARTERS = {"who", "what", "when", "where", "which", "name", "list", "define"}
_COMPLEX_KEYWORDS = {"how", "why", "explain", "describe", "compare", "analyze",
                     "evaluate", "discuss", "elaborate", "detail"}
_CROSS_SECTION_KEYWORDS = {"and", "vs", "versus", "between", "relationship",
                           "difference", "similarities", "contrast", "both"}


def classify_question(query: str) -> str:
    """
    Classifies a query into one of three categories to guide strategy selection:

    - "short_factual" — Short queries seeking a specific fact (who/what/when/where)
    - "complex"       — Longer queries requiring explanation or analysis
    - "cross_section" — Queries spanning multiple topics or documents

    The classification drives the DECIDE stage: different question types
    respond best to different chunk sizes and retrieval strategies.

    Args:
        query: The user's question.

    Returns:
        One of: "short_factual", "complex", "cross_section"
    """
    if not query:
        return "short_factual"

    words = query.lower().split()
    word_set = set(words)
    num_words = len(words)

    # Check for cross-section indicators (multi-topic queries)
    cross_section_matches = word_set & _CROSS_SECTION_KEYWORDS
    if len(cross_section_matches) >= 1 and num_words > 8:
        return "cross_section"

    # Check for complex query indicators
    complex_matches = word_set & _COMPLEX_KEYWORDS
    if complex_matches and num_words > 12:
        return "complex"

    # Short factual: short queries or starting with factual keywords
    if num_words <= 10 or (words[0] in _FACTUAL_STARTERS):
        return "short_factual"

    # Default: if > 15 words and has complex keywords, it's complex
    if num_words > 15 and complex_matches:
        return "complex"

    return "short_factual"
