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


def update_log_eval_metrics(log_id: int, answer_sem_sim: float, ctx_q_sim: float = None):
    """
    Updates a query log row with evaluation metrics computed by run_evaluation.py.
    Called after answer_query() when ground truth is available.
    """
    if log_id < 0:
        return
    session = get_session()
    try:
        log = session.query(QueryLog).filter(QueryLog.id == log_id).first()
        if log:
            log.answer_sem_sim = round(answer_sem_sim, 4)
            if ctx_q_sim is not None:
                log.ctx_q_sim = round(ctx_q_sim, 4)
            session.commit()
    except Exception as e:
        session.rollback()
        print(f"[logger] non-fatal — could not update eval metrics: {e}")
    finally:
        session.close()
