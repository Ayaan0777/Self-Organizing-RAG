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
from config import settings


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
    threshold: float = None,
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
    if threshold is None:
        threshold = settings.precision_relevance_threshold

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
#  1b. RETRIEVAL RECALL @ K
# ══════════════════════════════════════════════════════════════

def retrieval_recall_at_k(
    chunks: list[str],
    ground_truths: list[str],
    threshold: float = None,
) -> float:
    """
    Measures what fraction of ground-truth answers are covered by at least
    one retrieved chunk.

    For each ground-truth, we compute its semantic similarity to every chunk.
    A ground-truth is "recalled" if its best chunk match ≥ threshold.

    This is the dual of precision@K:
      - Precision: "Are the retrieved chunks relevant?"
      - Recall:    "Did we retrieve everything that's relevant?"

    Args:
        chunks: List of retrieved chunk text strings (top-K).
        ground_truths: List of reference answer strings.
        threshold: Minimum similarity to count as recalled.

    Returns:
        Recall score (0.0–1.0). E.g., 0.5 means half the ground truths
        are covered by at least one retrieved chunk.
    """
    if threshold is None:
        threshold = settings.precision_relevance_threshold

    if not chunks or not ground_truths:
        return 0.0

    try:
        chunk_embeddings = [_get_embedding(chunk) for chunk in chunks]
        recalled_count = 0

        for gt in ground_truths:
            gt_emb = _get_embedding(gt)
            # Check if ANY chunk covers this ground truth
            best_sim = max(
                _cosine_sim(gt_emb, chunk_emb) for chunk_emb in chunk_embeddings
            )
            if best_sim >= threshold:
                recalled_count += 1

        return round(recalled_count / len(ground_truths), 4)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════
#  2. CONTEXT SUFFICIENCY
# ══════════════════════════════════════════════════════════════

def context_sufficiency(
    chunks: list[str],
    ground_truths: list[str],
    threshold: float = None,
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
    if threshold is None:
        threshold = settings.sufficiency_threshold

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
    threshold: float = None,
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
    if threshold is None:
        threshold = settings.hallucination_grounding_threshold

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
#  4. QUESTION CLASSIFIER — Embedding-based (NLP)
# ══════════════════════════════════════════════════════════════

# Prototype queries for each category — the classifier compares incoming
# queries against these prototypes using cosine similarity.
# To tune classification, add/edit/remove prototype sentences.
_CATEGORY_PROTOTYPES = {
    "short_factual": [
        "What year did this event happen?",
        "Who invented the telephone?",
        "Name the capital of France",
        "When was the company founded?",
        "Where is the headquarters located?",
        "Which element has atomic number 6?",
        "What is the definition of osmosis?",
        "List the main components",
        "How many people were involved?",
        "What does this term mean?",
    ],
    "complex": [
        "Explain how photosynthesis works in detail",
        "Describe the process of cellular respiration and its stages",
        "Why did the Roman Empire decline and fall?",
        "Analyze the impact of climate change on agriculture",
        "How does machine learning differ from traditional programming and what are the implications?",
        "Discuss the advantages and disadvantages of renewable energy",
        "Elaborate on the causes and consequences of inflation",
        "Evaluate the effectiveness of this policy",
        "What are the underlying mechanisms behind this phenomenon?",
        "Provide a detailed explanation of how this system operates",
    ],
    "cross_section": [
        "Compare X and Y across multiple dimensions",
        "What is the relationship between A and B?",
        "How do these two approaches differ from each other?",
        "Contrast the economic policies of these two countries",
        "What are the similarities and differences between mitosis and meiosis?",
        "How does factor A interact with factor B?",
        "Describe the connection between poverty and education",
        "What links these two historical events?",
        "Compare the performance of method A versus method B",
        "How are these concepts related to each other?",
    ],
}

# Cache for prototype embeddings — computed lazily on first call
_prototype_cache: dict[str, list[np.ndarray]] = {}


def _get_prototype_embeddings() -> dict[str, list[np.ndarray]]:
    """
    Lazily computes and caches prototype embeddings.
    Called once on first classify_question() invocation, then reused.
    """
    global _prototype_cache
    if _prototype_cache:
        return _prototype_cache

    try:
        for category, prototypes in _CATEGORY_PROTOTYPES.items():
            _prototype_cache[category] = [
                _get_embedding(proto) for proto in prototypes
            ]
    except Exception:
        _prototype_cache = {}  # reset on failure

    return _prototype_cache


def classify_question(query: str) -> str:
    """
    Classifies a query into one of three categories using embedding similarity:

    - "short_factual" — Short queries seeking a specific fact
    - "complex"       — Queries requiring explanation or analysis
    - "cross_section" — Queries spanning multiple topics or comparisons

    Method:
      1. Embed the incoming query using mxbai-embed-large
      2. Compare against pre-embedded prototype queries for each category
      3. Compute mean similarity to each category's prototypes
      4. Return the category with the highest mean similarity

    Falls back to keyword-based heuristic if embedding fails.

    Args:
        query: The user's question.

    Returns:
        One of: "short_factual", "complex", "cross_section"
    """
    if not query:
        return "short_factual"

    try:
        proto_embeds = _get_prototype_embeddings()
        if not proto_embeds:
            return _classify_question_fallback(query)

        query_emb = _get_embedding(query)

        best_category = "short_factual"
        best_score = -1.0

        for category, embeddings in proto_embeds.items():
            # Mean similarity to all prototypes in this category
            sims = [_cosine_sim(query_emb, proto_emb) for proto_emb in embeddings]
            mean_sim = float(np.mean(sims))

            if mean_sim > best_score:
                best_score = mean_sim
                best_category = category

        return best_category

    except Exception:
        return _classify_question_fallback(query)


# ── Keyword fallback (used when embedding model is unavailable) ──
_FACTUAL_STARTERS = {"who", "what", "when", "where", "which", "name", "list", "define"}
_COMPLEX_KEYWORDS = {"how", "why", "explain", "describe", "compare", "analyze",
                     "evaluate", "discuss", "elaborate", "detail"}
_CROSS_SECTION_KEYWORDS = {"and", "vs", "versus", "between", "relationship",
                           "difference", "similarities", "contrast", "both"}


def _classify_question_fallback(query: str) -> str:
    """Keyword-based fallback when embedding model is unavailable."""
    words = query.lower().split()
    word_set = set(words)
    num_words = len(words)

    cross_section_matches = word_set & _CROSS_SECTION_KEYWORDS
    if len(cross_section_matches) >= 1 and num_words > 8:
        return "cross_section"

    complex_matches = word_set & _COMPLEX_KEYWORDS
    if complex_matches and num_words > 12:
        return "complex"

    if num_words <= 10 or (words[0] in _FACTUAL_STARTERS):
        return "short_factual"

    if num_words > 15 and complex_matches:
        return "complex"

    return "short_factual"

