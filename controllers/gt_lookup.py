"""
Ground Truth Lookup — Auto-RAG Pipeline
========================================
Loads dataset/long_ans.json on first access and provides fast lookup by
question text. When a user-typed query matches a known dataset entry,
controllers/retrieval.answer_query() enriches the QueryLog row with
GT-backed metrics (precision, sufficiency, hallucination, semantic
similarity) — same metrics /evaluate-local computes in batch mode.

Matching is normalized exact-match (lowercase, whitespace collapsed,
trailing punctuation stripped). Paraphrases won't match — add an
embedding-based fallback if that becomes needed.

To refresh after editing long_ans.json without restarting uvicorn,
call reload_dataset() (e.g. via a debug endpoint).
"""
import json
import logging
import os
import re
from typing import Optional

_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "dataset", "long_ans.json",
)

_gt_index: dict = {}   # normalized question → list[str] ground truths
_loaded: bool = False


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation."""
    t = text.lower().strip()
    t = re.sub(r"[?.!,;:]+$", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _load_dataset():
    """Loads long_ans.json into the in-memory index. Idempotent."""
    global _gt_index, _loaded
    if _loaded:
        return
    try:
        if not os.path.exists(_DATASET_PATH):
            logging.warning(f"[gt_lookup] dataset not found at {_DATASET_PATH}")
            _loaded = True
            return
        with open(_DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            q = item.get("qun", "")
            a = item.get("ans", [])
            if q and a:
                _gt_index[_normalize(q)] = a if isinstance(a, list) else [a]
        logging.info(f"[gt_lookup] loaded {len(_gt_index)} known questions from long_ans.json")
        print(f"[gt_lookup] loaded {len(_gt_index)} known questions from long_ans.json")
    except Exception as e:
        logging.warning(f"[gt_lookup] failed to load dataset: {e}")
    finally:
        _loaded = True


def lookup_ground_truth(query: str) -> Optional[list]:
    """
    Returns the ground-truth answers list for a query, or None if no match.
    Matches are normalized exact (case + whitespace + trailing punctuation).
    """
    _load_dataset()
    return _gt_index.get(_normalize(query))


def reload_dataset():
    """Force a fresh read of long_ans.json (after editing the file on disk)."""
    global _gt_index, _loaded
    _gt_index = {}
    _loaded = False
    _load_dataset()


def enrich_log_with_gt(
    log_id: int,
    query: str,
    answer: str,
    contexts: list,
    gts: list,
):
    """
    Computes GT-backed metrics for a freshly-logged query and persists
    them to the QueryLog row. Mirrors what process_local_evaluation()
    computes per-question, just for the single ad-hoc query.

    Stage 1 metrics: answer_sem_sim, ctx_q_sim, retrieved_contexts
    Stage 2 metrics: retrieval_precision, context_sufficiency,
                     hallucination_rate, question_category
    """
    # Lazy imports — these modules depend on controllers/retrieval, so
    # importing at module top would create a cycle.
    from controllers.evaluation import semantic_similarity, context_similarity
    from controllers.metrics import (
        retrieval_precision_at_k,
        context_sufficiency as compute_context_sufficiency,
        hallucination_rate as compute_hallucination_rate,
        classify_question,
    )
    from logger.query_logger import (
        update_log_eval_metrics,
        update_log_new_metrics,
    )

    # Stage 1: answer similarity + context-question similarity
    ss = max((semantic_similarity(answer, gt) for gt in gts), default=0.0)
    ctx_sims = context_similarity(query, gts, contexts)

    # Stage 2: retrieval-quality + grounding metrics
    ret_precision = retrieval_precision_at_k(contexts, gts)
    ctx_suff = compute_context_sufficiency(contexts, gts)
    hall_rate = compute_hallucination_rate(answer, contexts)
    q_category = classify_question(query)

    print(
        f"[gt_lookup] enriched log_id={log_id} | "
        f"sim={ss:.3f} precision={ret_precision:.3f} "
        f"sufficiency={ctx_suff} hallucination={hall_rate:.3f} "
        f"category={q_category}"
    )

    update_log_eval_metrics(
        log_id=log_id,
        answer_sem_sim=ss,
        ctx_q_sim=ctx_sims["ctx_question_sim"],
        retrieved_contexts=contexts,
    )
    update_log_new_metrics(
        log_id=log_id,
        retrieval_precision=ret_precision,
        context_sufficiency=ctx_suff,
        hallucination_rate=hall_rate,
        question_category=q_category,
    )
