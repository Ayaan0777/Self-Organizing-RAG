"""
LLM-Based Reranker — Phase 2: Improved Retrieval
==================================================
Re-scores retrieved chunks using the LLM's deeper understanding
of relevance, then reorders them. Uses a single batch LLM call
(not one call per chunk) so the overhead is manageable.

Safety: Falls back to original order on any parse/LLM failure.
Can be disabled by passing rerank=False to answer_query().
"""
import json
import re
import logging
from langchain_core.documents import Document


def rerank_chunks(
    query: str,
    docs: list[Document],
    scores: list[float],
) -> tuple[list[Document], list[float]]:
    """
    Uses the LLM to re-rank retrieved chunks by relevance to the query.

    The LLM receives all chunks at once and returns a JSON array of
    chunk numbers ordered from most to least relevant. One LLM call total.

    Args:
        query:  The user's query string.
        docs:   Retrieved Documents from Pinecone.
        scores: Cosine similarity scores corresponding to each doc.

    Returns:
        (reordered_docs, reordered_scores)
        On failure, returns the original (docs, scores) unchanged.
    """
    if len(docs) <= 1:
        return docs, scores

    try:
        from services.llm_factory import get_llm
        llm = get_llm()

        # Build the prompt with numbered chunks
        chunks_text = ""
        for i, doc in enumerate(docs):
            # Truncate to avoid overwhelming the LLM
            preview = doc.page_content[:300].replace("\n", " ")
            chunks_text += f"[{i + 1}] {preview}\n\n"

        prompt = (
            f"Given this search query: \"{query}\"\n\n"
            f"Rank these {len(docs)} text chunks from MOST to LEAST relevant "
            f"to the query. Return ONLY a JSON array of the chunk numbers "
            f"in order of relevance.\n\n"
            f"{chunks_text}"
            f"Return ONLY a JSON array like [{', '.join(str(i+1) for i in range(len(docs)))}]:"
        )

        raw_response = llm.invoke(prompt).content.strip()

        # Extract JSON array from response (LLM might wrap it in text)
        ranking = _parse_ranking(raw_response, len(docs))

        if ranking is None:
            logging.warning("[reranker] could not parse LLM ranking — keeping original order")
            return docs, scores

        # Reorder docs and scores based on ranking
        reordered_docs = []
        reordered_scores = []
        for idx in ranking:
            reordered_docs.append(docs[idx])
            reordered_scores.append(scores[idx])

        logging.info(
            f"[reranker] reranked {len(docs)} chunks: "
            f"original top={scores[0]:.4f}, reranked top={reordered_scores[0]:.4f}"
        )
        return reordered_docs, reordered_scores

    except Exception as e:
        logging.warning(f"[reranker] failed ({e}), keeping original order")
        return docs, scores


def _parse_ranking(raw_response: str, num_chunks: int) -> list[int] | None:
    """
    Extracts a valid ranking from the LLM's response.

    Handles edge cases:
      - JSON wrapped in markdown code blocks
      - Text before/after the JSON
      - Out-of-range indices
      - Duplicate indices
      - Fewer indices than expected

    Returns:
        List of 0-indexed chunk indices in ranked order, or None on failure.
    """
    try:
        # Try to find a JSON array in the response
        # Match [...] pattern, allowing for text before/after
        match = re.search(r'\[[\d,\s]+\]', raw_response)
        if not match:
            return None

        ranking_1indexed = json.loads(match.group())

        # Validate and convert to 0-indexed
        seen = set()
        ranking_0indexed = []

        for idx in ranking_1indexed:
            if not isinstance(idx, int):
                continue
            zero_idx = idx - 1  # convert 1-indexed to 0-indexed
            if 0 <= zero_idx < num_chunks and zero_idx not in seen:
                seen.add(zero_idx)
                ranking_0indexed.append(zero_idx)

        if not ranking_0indexed:
            return None

        # Append any missing indices at the end (in original order)
        for i in range(num_chunks):
            if i not in seen:
                ranking_0indexed.append(i)

        return ranking_0indexed

    except (json.JSONDecodeError, TypeError, ValueError):
        return None
