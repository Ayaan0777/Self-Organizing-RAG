"""
Retrieval Gate — Self-Organising RAG
=====================================
Decides WHETHER a query needs document retrieval at all.

Queries like "hello", "thanks", or "what is 2+2" don't need Pinecone.
Factual/domain queries like "What is binary search?" do need retrieval.

Safety: Always falls back to RETRIEVE on any error — never blocks a query.
"""
import logging

_DIRECT_ANSWER_PATTERNS = [
    "hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye",
    "good morning", "good evening", "good night", "how are you",
    "what's up", "ok", "okay", "yes", "no", "sure", "please",
]


def check_retrieval_needed(query: str) -> dict:
    """
    Determines if a query requires document retrieval or can be answered directly.

    Uses a two-tier approach:
      1. Fast pattern matching for obvious greetings/chitchat
      2. LLM classification for ambiguous cases

    Args:
        query: The user's raw query string.

    Returns:
        {
            "needs_retrieval": bool,
            "reason": str,       # "pattern_match" | "llm_classify" | "fallback"
            "gate_detail": str,  # human-readable explanation
        }
    """
    cleaned = query.strip().lower().rstrip("?!.,")

    # ── Tier 1: Fast pattern match for obvious non-retrieval queries ──
    if cleaned in _DIRECT_ANSWER_PATTERNS:
        return {
            "needs_retrieval": False,
            "reason": "pattern_match",
            "gate_detail": f"Query '{cleaned}' matched a known conversational pattern.",
        }

    # Very short queries without question words are likely chitchat
    words = cleaned.split()
    if len(words) <= 2 and not any(w in words for w in ["what", "how", "why", "when", "where", "who", "which", "explain", "describe", "define"]):
        return {
            "needs_retrieval": False,
            "reason": "pattern_match",
            "gate_detail": "Short non-question query — likely conversational.",
        }

    # ── Tier 2: LLM classification for ambiguous queries ──
    try:
        from services.llm_factory import get_llm
        llm = get_llm()

        prompt = (
            "Classify this query. Reply with ONLY the word RETRIEVE or DIRECT.\n"
            "- RETRIEVE: query asks about specific facts, documents, data, or domain knowledge.\n"
            "- DIRECT: query is a greeting, chitchat, math, or general knowledge the LLM knows.\n\n"
            f"Query: {query}\n\n"
            "Classification:"
        )

        response = llm.invoke(prompt).content.strip().upper()

        if "DIRECT" in response:
            return {
                "needs_retrieval": False,
                "reason": "llm_classify",
                "gate_detail": "LLM classified query as answerable without retrieval.",
            }

        # "RETRIEVE" or anything else → retrieve
        return {
            "needs_retrieval": True,
            "reason": "llm_classify",
            "gate_detail": "LLM classified query as requiring document retrieval.",
        }

    except Exception as e:
        logging.warning(f"[retrieval_gate] classification failed ({e}), defaulting to RETRIEVE")
        return {
            "needs_retrieval": True,
            "reason": "fallback",
            "gate_detail": f"Gate classification failed — defaulting to retrieval. Error: {e}",
        }


# Note: the higher-level fallback dict built in controllers/retrieval.py
# (when the gate import itself fails) also needs to include "gate_detail"
# for consistency. That's handled at the call site.
