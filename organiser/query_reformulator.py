"""
Query Reformulator — Phase 4: Self-Organising RAG
===================================================
When initial retrieval quality is poor (top-1 score below threshold),
asks the LLM to rephrase the query and retries retrieval.

Safety:
  - Max 1 reformulation attempt (no infinite loops)
  - Falls back to original results on any failure
  - Only replaces results if the rephrased query scores genuinely better
"""
import logging


REFORMULATION_THRESHOLD = 0.45  # same as low_top_score detector threshold


def try_reformulate(query: str, vector_store, k: int, current_top_score: float) -> dict | None:
    """
    Attempts to improve retrieval by rephrasing the query.

    Called when the initial top-1 retrieval score is below REFORMULATION_THRESHOLD.
    Asks the LLM to rephrase, retries retrieval, and returns the better result set.

    Args:
        query:              The original user query.
        vector_store:       LangChain PineconeVectorStore instance.
        k:                  Number of chunks to retrieve.
        current_top_score:  The top-1 score from the initial retrieval.

    Returns:
        If improvement found:
            {
                "docs": list[Document],
                "scores": list[float],
                "reformulated_query": str,
            }
        If no improvement or failure:
            None  (caller should use original results)
    """
    if current_top_score >= REFORMULATION_THRESHOLD:
        return None  # score is fine, no reformulation needed

    try:
        from services.llm_factory import get_llm
        llm = get_llm()

        prompt = (
            "The following search query returned poor results from a document database. "
            "Rephrase it to be more specific and likely to match relevant document content. "
            "Return ONLY the rephrased query string, nothing else.\n\n"
            f"Original query: {query}\n\n"
            "Rephrased query:"
        )

        raw_response = llm.invoke(prompt).content.strip()

        # Clean up: remove quotes, extra whitespace
        rephrased = raw_response.strip('"').strip("'").strip()

        if not rephrased or rephrased.lower() == query.lower():
            logging.info("[reformulator] LLM returned same/empty query — skipping")
            return None

        logging.info(f"[reformulator] original: '{query}' → rephrased: '{rephrased}'")

        # Retry retrieval with the rephrased query
        new_results = vector_store.similarity_search_with_score(rephrased, k=k)

        if not new_results:
            return None

        new_docs = [d for d, _ in new_results]
        new_scores = [round(float(s), 4) for _, s in new_results]

        # Only use rephrased results if they actually improve the top score
        if new_scores[0] > current_top_score:
            improvement = new_scores[0] - current_top_score
            logging.info(
                f"[reformulator] ✨ improvement! "
                f"score {current_top_score:.4f} → {new_scores[0]:.4f} (+{improvement:.4f})"
            )
            return {
                "docs": new_docs,
                "scores": new_scores,
                "reformulated_query": rephrased,
            }
        else:
            logging.info(
                f"[reformulator] rephrased query did not improve "
                f"(original {current_top_score:.4f} vs rephrased {new_scores[0]:.4f})"
            )
            return None

    except Exception as e:
        logging.warning(f"[reformulator] failed ({e}), keeping original results")
        return None
