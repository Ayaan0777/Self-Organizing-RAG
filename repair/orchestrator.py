"""
Repair Orchestrator — SH2: DECIDE → ACT → PROBE → COMMIT/ROLLBACK
===================================================================
Implements the mentor's Stage 3 (DECIDE) and Stage 4 (ACT) pipeline.

When a low-recall event is triggered:
  1. DECIDE: Select repair chunk_size based on failure pattern + query complexity
  2. BACKUP: Fetch old vectors from Pinecone into memory
  3. ACT: Delete old chunks → rechunk with DECIDED params → insert new chunks
  4. PROBE: Re-run original query → check if score improved
  5. COMMIT or ROLLBACK:
     - Improved → keep new chunks, discard backup
     - NOT improved → delete new chunks, restore old from backup

SAFETY: Old data is NEVER permanently lost. Rollback restores exact original state.
"""
import json
import time
import logging

from db.session import get_session
from db.models import LowRecallEvent, QueryLog, RepairReport
from repair.chunker import rechunk_adaptive, select_repair_params
from repair.reembedder import (
    backup_chunks,
    delete_chunks,
    insert_chunks,
    rollback,
    probe_score,
)
from services.llm_factory import get_pinecone_index, get_embeddings
from config import settings

# Score improvement threshold — repair must beat original by at least this much
IMPROVEMENT_THRESHOLD = 0.05


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
    texts = []
    source = "unknown"
    for m in results.matches:
        text = m.metadata.get("text", "")
        if not text:
            text = m.metadata.get("page_content", "")
        texts.append(text)
        if m.metadata.get("source"):
            source = m.metadata["source"]

    full_text = " ".join(texts)
    return chunk_ids, full_text, source


def handle_event(event_id: int) -> dict:
    """
    Full repair pipeline for one LowRecallEvent:

      DECIDE → BACKUP → DELETE → RECHUNK → INSERT → PROBE → COMMIT/ROLLBACK → LOG

    The repair strategy (chunk_size) is automatically selected based on:
      1. Which detectors triggered (context_insufficient, hallucination_detected, etc.)
      2. Query complexity (complex → bigger chunks, simple → smaller chunks)

    Rollback ensures old data is restored if repair doesn't improve results.
    """
    session = get_session()
    t0 = time.time()

    try:
        # ── Load event + query ──────────────────────────────────────
        event = session.query(LowRecallEvent).filter(
            LowRecallEvent.id == event_id
        ).first()

        if not event:
            return {"error": f"Event {event_id} not found"}
        if event.resolved:
            return {"message": f"Event {event_id} is already resolved — skipping"}

        log = session.query(QueryLog).filter(
            QueryLog.id == event.query_log_id
        ).first()

        if not log:
            return {"error": "Original query log entry missing"}

        scores_list = json.loads(log.top_k_scores or "[0]")
        score_before = scores_list[0] if scores_list else 0.0
        detectors = json.loads(event.triggered_detectors or "[]")
        ns = settings.pinecone_namespace

        # ── DECIDE: Select repair params ────────────────────────────
        params = select_repair_params(log.query, detectors)
        logging.info(
            f"[repair] DECIDE: event={event_id} reason={params['reason']} "
            f"chunk_size={params['chunk_size']} overlap={params['overlap']}"
        )

        # ── Get old chunk IDs + text ────────────────────────────────
        chunk_ids, full_text, source = _get_chunk_ids_for_query(log.query, namespace=ns)

        if not full_text.strip():
            return {"error": "Could not retrieve chunk text for repair — "
                             "chunks may not have text metadata stored"}
        if not chunk_ids:
            return {"error": "No chunks found for this query — ingest documents first"}

        logging.info(f"[repair] Found {len(chunk_ids)} old chunks to repair for event {event_id}")

        # Log old chunk sizes for comparison
        old_sizes = [len(t) for t in full_text.split(" ") if t.strip()] if full_text else []
        logging.info(
            f"[repair] OLD: {len(chunk_ids)} chunks, "
            f"total text length={len(full_text)} chars"
        )

        # ── BACKUP: Save old vectors (for rollback) ─────────────────
        backup_data = backup_chunks(chunk_ids, namespace=ns)

        # ── ACT: Delete old → Rechunk → Insert new ─────────────────
        delete_chunks(chunk_ids, namespace=ns)

        new_chunks = rechunk_adaptive(
            text=full_text,
            source=source,
            chunk_size=params["chunk_size"],
            overlap=params["overlap"],
            repair_reason=params["reason"],
        )

        # ── RECHUNKING VERIFICATION ─────────────────────────────────
        new_sizes = [len(c.page_content) for c in new_chunks]
        logging.info(
            f"[repair] RECHUNK VERIFICATION: "
            f"{len(chunk_ids)} old chunks → {len(new_chunks)} new chunks | "
            f"chunk_size={params['chunk_size']} (ingestion was 1250) | "
            f"new sizes: {new_sizes}"
        )
        new_ids = insert_chunks(new_chunks, namespace=ns)

        # ── PROBE: Check if repair improved ─────────────────────────
        score_after = probe_score(log.query, namespace=ns)
        improved = score_after > score_before + IMPROVEMENT_THRESHOLD

        # ── COMMIT or ROLLBACK ──────────────────────────────────────
        if improved:
            # COMMIT: new chunks stay, backup discarded
            event.resolved = True
            logging.info(
                f"[repair] ✅ COMMITTED event={event_id}: "
                f"{score_before:.3f} → {score_after:.3f} (+{score_after - score_before:.3f})"
            )
        else:
            # ROLLBACK: remove new chunks, restore old from backup
            rollback_ok = rollback(backup_data, new_ids, namespace=ns)
            event.unfixable = True
            logging.info(
                f"[repair] 🔄 ROLLED BACK event={event_id}: "
                f"{score_before:.3f} → {score_after:.3f} "
                f"(no improvement, rollback={'OK' if rollback_ok else 'FAILED'})"
            )

        # ── LOG: Write RepairReport ─────────────────────────────────
        duration_ms = int((time.time() - t0) * 1000)

        report = RepairReport(
            event_id        = event_id,
            strategy_used   = params["reason"],
            chunk_size_used = params["chunk_size"],
            repair_reason   = params["reason"],
            chunks_before   = len(chunk_ids),
            chunks_after    = len(new_chunks),
            score_before    = round(score_before, 4),
            score_after     = round(score_after, 4),
            resolved        = improved,
            rolled_back     = not improved,
            duration_ms     = duration_ms,
        )
        session.add(report)
        session.commit()

        return {
            "event_id":      event_id,
            "strategy":      params["reason"],
            "chunk_size":    params["chunk_size"],
            "score_before":  round(score_before, 4),
            "score_after":   round(score_after, 4),
            "improved":      improved,
            "rolled_back":   not improved,
            "chunks_before": len(chunk_ids),
            "chunks_after":  len(new_chunks),
            "duration_ms":   duration_ms,
        }

    except Exception as e:
        session.rollback()
        logging.error(f"[repair] event={event_id} failed: {e}")
        return {"error": str(e)}
    finally:
        session.close()
