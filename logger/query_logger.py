import json
from db.session import get_session
from db.models import QueryLog


def log_query(
    query:           str,
    scores:          list,
    answer:          str,
    latency_ms:      int,
    chunk_metadatas: list = None,
    chunk_contents:  list = None,
) -> int:
    """
    Persists one query event to autorag_query_log.
    Returns the new row id, or -1 if writing fails (non-fatal).
    """
    session = get_session()
    try:
        ids = []
        if chunk_metadatas:
            for m in chunk_metadatas:
                ids.append(m.get("id") or str(m.get("source", ""))[:60])

        # ctx_q_sim = average of top-k scores (scores are already cosine similarities)
        ctx_q_sim = round(sum(scores) / len(scores), 4) if scores else None

        entry = QueryLog(
            query            = query,
            chunk_ids        = json.dumps(ids),
            top_k_scores     = json.dumps([round(float(s), 4) for s in scores]),
            retrieved_chunks = json.dumps(chunk_contents or [], ensure_ascii=False),
            llm_response     = answer,
            latency_ms       = latency_ms,
            ctx_q_sim        = ctx_q_sim,
            answer_sem_sim   = None,   # set later by run_evaluation.py if ground truth exists
            flagged          = False,
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return entry.id

    except Exception as e:
        session.rollback()
        print(f"[logger] non-fatal — could not write query log: {e}")
        return -1
    finally:
        session.close()

def update_log_eval_metrics(log_id: int, answer_sem_sim: float, ctx_q_sim: float):
    """
    Writes answer-similarity + context-question similarity onto a QueryLog row.
    The retrieved chunk texts are already persisted by log_query() into the
    retrieved_chunks column — there was a defunct retrieved_contexts kwarg here
    that wrote to a non-existent column and got silently dropped. Removed.
    """
    session = get_session()
    try:
        log_entry = session.query(QueryLog).filter(QueryLog.id == log_id).first()
        if log_entry:
            log_entry.answer_sem_sim = answer_sem_sim
            log_entry.ctx_q_sim = ctx_q_sim
            session.commit()
            print(f"      [db] updated evaluation metrics for log_id {log_id}")
    except Exception as e:
        session.rollback()
        print(f"      [db] failed to update metrics: {e}")
    finally:
        session.close()


def update_log_new_metrics(
    log_id: int,
    retrieval_precision: float = None,
    context_sufficiency: bool = None,
    hallucination_rate: float = None,
    question_category: str = None,
):
    """
    Persists the new Stage 2 metrics to an existing QueryLog row.
    Called by evaluation.py after computing retrieval precision,
    context sufficiency, hallucination rate, and question category.
    """
    session = get_session()
    try:
        log_entry = session.query(QueryLog).filter(QueryLog.id == log_id).first()
        if log_entry:
            if retrieval_precision is not None:
                log_entry.retrieval_precision = retrieval_precision
            if context_sufficiency is not None:
                log_entry.context_sufficiency = context_sufficiency
            if hallucination_rate is not None:
                log_entry.hallucination_rate = hallucination_rate
            if question_category is not None:
                log_entry.question_category = question_category

            session.commit()
            print(f"      [db] updated Stage 2 metrics for log_id {log_id}")
    except Exception as e:
        session.rollback()
        print(f"      [db] failed to update Stage 2 metrics: {e}")
    finally:
        session.close()