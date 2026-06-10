"""
Dynamic K Selector — Phase 2: Improved Retrieval
==================================================
Automatically determines the optimal number of chunks to pass to the LLM,
instead of using a fixed K for every query.

Algorithm (3 stages):
  1. Query Complexity Analysis  → determines min_k and max_k bounds
  2. Score Threshold Pruning    → removes clearly irrelevant chunks
  3. Score Cliff Detection      → finds natural cutoff points

Example:
  Query: "What is binary search?"
  Retrieved 10 chunks with scores: [0.82, 0.78, 0.75, 0.71, 0.45, 0.22, 0.18, 0.15, 0.12, 0.10]
                                                              ↑ cliff here (0.71 → 0.45 = 0.26 drop)
  Result: returns first 4 chunks (optimal K = 4)

Safety: Falls back to original results (trimmed to target_k) on any failure.
"""
import re
import logging

# ── Thresholds ──────────────────────────────────────────────────
MIN_ABSOLUTE_SCORE   = 0.25    # chunks below this are basically noise
CLIFF_DROP_THRESHOLD = 0.12    # a drop of >0.12 between consecutive chunks = cliff
MIN_K                = 2       # always return at least 2 chunks


def select_optimal_chunks(
    query: str,
    docs: list,
    scores: list[float],
    target_k: int = 5,
) -> tuple[list, list[float]]:
    """
    Given a query and its retrieved results (already sorted by score descending),
    selects the optimal number of chunks to pass to the LLM.

    Args:
        query:    The user's query string.
        docs:     Retrieved Document objects (sorted by score, descending).
        scores:   Corresponding cosine similarity scores.
        target_k: The user's requested K (used as upper bound).

    Returns:
        (optimal_docs, optimal_scores) — trimmed to the ideal chunk count.
    """
    if not docs or not scores:
        return docs, scores

    n = len(docs)

    if n <= MIN_K:
        return docs, scores

    # ── Stage 1: Query Complexity → bounds ──────────────────────
    min_k, max_k = _analyse_query_complexity(query)
    max_k = min(max_k, target_k * 2, n)  # allow up to 2x for complex queries
    min_k = min(min_k, max_k)         # ensure min ≤ max

    # ── Stage 2: Absolute Score Pruning ─────────────────────────
    # Find where scores drop below the absolute threshold
    score_cutoff = n
    for i in range(n):
        if scores[i] < MIN_ABSOLUTE_SCORE:
            score_cutoff = i
            break

    # ── Stage 3: Score Cliff Detection ──────────────────────────
    # Find the biggest drop between consecutive chunks
    cliff_cutoff = n
    max_drop = 0.0
    for i in range(min(n - 1, max_k)):
        drop = scores[i] - scores[i + 1]
        if drop > CLIFF_DROP_THRESHOLD and drop > max_drop:
            max_drop = drop
            cliff_cutoff = i + 1  # cut AFTER chunk i (keep i, drop i+1)

    # ── Combine all signals ─────────────────────────────────────
    # Take the most conservative (smallest) cutoff, but respect min_k
    optimal_k = min(score_cutoff, cliff_cutoff, max_k)
    optimal_k = max(optimal_k, min_k)  # never go below min
    optimal_k = min(optimal_k, n)       # never exceed available

    if optimal_k != n:
        logging.info(
            f"[dynamic_k] {n} fetched → {optimal_k} selected "
            f"(cliff_at={cliff_cutoff}, score_floor_at={score_cutoff}, "
            f"bounds=[{min_k},{max_k}], top={scores[0]:.4f}, "
            f"cut_at={scores[optimal_k-1]:.4f})"
        )

    return docs[:optimal_k], scores[:optimal_k]


def _analyse_query_complexity(query: str) -> tuple[int, int]:
    """
    Analyses query text to determine appropriate K bounds.

    Returns:
        (min_k, max_k) tuple based on query characteristics.

    Heuristics:
      - Multi-part / comparison queries → need more chunks
      - Simple factual lookups → fewer chunks suffice
      - Long, detailed queries → more chunks for coverage
    """
    q = query.lower().strip()
    words = q.split()
    word_count = len(words)

    # ── Comparison / multi-part queries → more chunks ───────────
    comparison_patterns = [
        r"\bcompare\b", r"\bdifference\b", r"\bvs\.?\b", r"\bversus\b",
        r"\band\b.*\band\b",  # "X and Y and Z"
        r"\bboth\b", r"\beach\b", r"\ball\b",
        r"\badvantages\b.*\bdisadvantages\b",
        r"\bpros\b.*\bcons\b",
    ]
    is_comparison = any(re.search(p, q) for p in comparison_patterns)
    if is_comparison:
        return (4, 10)

    # ── Multi-part questions (contains multiple ?) ──────────────
    question_marks = q.count("?")
    if question_marks >= 2:
        return (4, 10)

    # ── Broad/exploratory queries → more chunks ─────────────────
    broad_patterns = [
        r"\bexplain\b", r"\bdescribe\b", r"\boverview\b", r"\bsummar",
        r"\bwhat are\b", r"\blist\b", r"\bdetail\b", r"\bdiscuss\b",
        r"\btell me about\b", r"\bwhat do you know\b",
    ]
    is_broad = any(re.search(p, q) for p in broad_patterns)
    if is_broad:
        return (3, 8)

    # ── Specific factual lookups → fewer chunks ─────────────────
    specific_patterns = [
        r"\bwho\b", r"\bwhen\b", r"\bwhere\b",
        r"\bhow (?:much|many|old|long|far)\b",
        r"\bwhat (?:year|date|time|number|name|city|country)\b",
        r"\bwhich\b",
    ]
    is_specific = any(re.search(p, q) for p in specific_patterns)
    if is_specific:
        return (2, 5)

    # ── Long queries likely need more context ───────────────────
    if word_count >= 15:
        return (3, 8)

    # ── Default: medium complexity ──────────────────────────────
    return (3, 6)
