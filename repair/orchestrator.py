"""
Repair Orchestrator — SAFE repair loop
=======================================
When a low-recall event is detected, this module:
  1. Loads the failing query from QueryLog
  2. Retrieves the poorly-matching chunks (with their Pinecone IDs)
  3. Rechunks the text with a better strategy
  4. Replaces ONLY those specific chunks (not the whole document)
  5. Probes recall to check if the repair improved results

SAFETY: Only the specific failing chunks are replaced. The rest of
the index (all other chunks from the same document) stays intact.
"""
import json
import time

from db.session import get_session
from db.models import LowRecallEvent, QueryLog, RepairReport
from repair.chunker import rechunk_semantic, rechunk_llm, rechunk_entropy
from repair.reembedder import reembed
from controllers.retrieval import generate_answer_only
from services.llm_factory import get_vector_store, get_pinecone_index, get_embeddings
from config import settings

STRATEGY_MAP = {
    "semantic": rechunk_semantic,
    "llm":      rechunk_llm,
    "entropy":  rechunk_entropy,
}


def _probe_score(query: str, namespace: str = None) -> float:
    """Re-runs the original failing query and returns the new top-1 score."""
    vs     = get_vector_store(namespace)
    result = vs.similarity_search_with_score(query, k=1)
    # Pinecone returns cosine similarity (1 = identical)
    return round(float(result[0][1]), 4) if result else 0.0


def _get_chunk_ids_for_query(query: str, namespace: str = None, k: int = 5) -> tuple:
    """
    Retrieves chunks matching the query and returns their Pinecone vector IDs
    along with the concatenated text for rechunking.

    Returns: (chunk_ids: list[str], full_text: str, source: str)
    """
    ns = namespace or settings.pinecone_namespace
    index = get_pinecone_index()
    embeddings = get_embeddings()

    # Embed the query and search Pinecone directly to get vector IDs
    query_emb = embeddings.embed_query(query)
    results = index.query(
        vector=query_emb,
        top_k=k,
        namespace=ns,
        include_metadata=True,
    )

    if not results.matches:
        return [], "", "unknown"

    chunk_ids = [m.id for m in results.matches]
    # Reconstruct the text from metadata
    texts = []
    source = "unknown"
    for m in results.matches:
        text = m.metadata.get("text", "")
        if not text:
            # LangChain stores text in the 'text' metadata field
            text = m.metadata.get("page_content", "")
        texts.append(text)
        if m.metadata.get("source"):
            source = m.metadata["source"]

    full_text = " ".join(texts)
    return chunk_ids, full_text, source


def handle_event(event_id: int, strategy: str = "semantic") -> dict:
    """
    SAFE repair loop for one LowRecallEvent:
      1. Load the event and its original failing query
      2. Find the specific chunk IDs that were retrieved (poorly) 
      3. Rechunk their text with a better strategy
      4. Delete ONLY those specific chunks + insert new ones
      5. Probe recall to check improvement
      6. Write RepairReport

    SAFETY: Only the specific failing chunks are replaced.
    The rest of the index stays intact.
    """
    session = get_session()
    t0      = time.time()
    try:
        event = session.query(LowRecallEvent).filter(
                    LowRecallEvent.id == event_id).first()
        if not event:
            return {"error": f"Event {event_id} not found"}
        if event.resolved:
            return {"message": f"Event {event_id} is already resolved — skipping"}

        log = session.query(QueryLog).filter(
                  QueryLog.id == event.query_log_id).first()
        if not log:
            return {"error": "Original query log entry missing"}

        score_before = json.loads(log.top_k_scores or "[0]")[0]

        # Get the SPECIFIC chunk IDs and text for the failing query
        chunk_ids, full_text, source = _get_chunk_ids_for_query(log.query)

        if not full_text.strip():
            return {"error": "Could not retrieve chunk text for repair — "
                             "chunks may not have text metadata stored"}

        if not chunk_ids:
            return {"error": "No chunks found for this query — ingest documents first"}

        print(f"[repair] Found {len(chunk_ids)} chunks to replace for event {event_id}")

        # Rechunk the text with the chosen strategy
        rechunk_fn = STRATEGY_MAP.get(strategy, rechunk_semantic)
        new_chunks = rechunk_fn(full_text, source)

        # Replace ONLY the specific chunks (safe — no full source delete)
        counts = reembed(new_chunks, source, old_chunk_ids=chunk_ids)

        # Probe: did the repair actually improve recall?
        score_after = _probe_score(log.query)
        improved    = score_after > score_before + 0.05
        resolved_answer = None

        if improved:
            repaired_result = generate_answer_only(log.query)
            resolved_answer = repaired_result.get("answer")

        # Persist repair report
        report = RepairReport(
            event_id      = event_id,
            strategy_used = strategy,
            chunks_before = counts["old_count"],
            chunks_after  = counts["new_count"],
            score_before  = round(score_before, 4),
            score_after   = round(score_after, 4),
            resolved      = improved,
            original_answer = log.llm_response,
            resolved_answer = resolved_answer,
            duration_ms   = int((time.time() - t0) * 1000),
        )
        event.resolved = improved
        session.add(report)
        session.commit()

        status = "RESOLVED" if improved else "UNRESOLVED"
        print(f"[repair] event={event_id} strategy={strategy} "
              f"score {score_before:.3f} -> {score_after:.3f} {status}")

        return {
            "event_id"     : event_id,
            "strategy"     : strategy,
            "score_before" : round(score_before, 4),
            "score_after"  : round(score_after, 4),
            "improved"     : improved,
            "chunks_before": counts["old_count"],
            "chunks_after" : counts["new_count"],
            "duration_ms"  : report.duration_ms,
        }

    except Exception as e:
        session.rollback()
        return {"error": str(e)}
    finally:
        session.close()
