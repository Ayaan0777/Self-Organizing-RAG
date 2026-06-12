"""
Repair Reembedder — SAFE partial re-embedding with ROLLBACK
=============================================================
Only replaces the specific chunks related to a failing query.
NEVER deletes all vectors for a source — that would wipe the entire document.

Stage 4 additions:
  - Snapshot: Before deleting old chunks, saves a copy to ChunkSnapshot table
  - Rollback: Can restore old chunks from snapshot if repair degrades metrics

Strategy:
  1. Save a snapshot of the old chunks (text + metadata) to SQLite
  2. Delete ONLY those specific IDs from Pinecone
  3. Insert new rechunked versions alongside the existing index
  4. If repair fails → rollback_from_snapshot() restores originals
"""
import json
from services.llm_factory import get_vector_store, get_pinecone_index, get_embeddings
from langchain_core.documents import Document
from config import settings


def _save_chunk_snapshot(chunk_ids: list[str], namespace: str, event_id: int):
    """
    Saves a copy of old chunks to the ChunkSnapshot table before deletion.
    This enables rollback if the repair degrades metrics.
    """
    from db.session import get_session
    from db.models import ChunkSnapshot

    ns = namespace or settings.pinecone_namespace
    index = get_pinecone_index()
    session = get_session()

    try:
        # Fetch vectors from Pinecone in batches
        for i in range(0, len(chunk_ids), 50):
            batch_ids = chunk_ids[i:i + 50]
            fetch_result = index.fetch(ids=batch_ids, namespace=ns)

            for vid, vec_data in fetch_result.vectors.items():
                text = vec_data.metadata.get("text", "")
                snapshot = ChunkSnapshot(
                    event_id=event_id,
                    vector_id=vid,
                    text=text,
                    metadata_json=json.dumps(dict(vec_data.metadata), default=str),
                    namespace=ns,
                )
                session.add(snapshot)

        session.commit()
        print(f"[reembedder] saved snapshot of {len(chunk_ids)} chunks for event {event_id}")
    except Exception as e:
        session.rollback()
        print(f"[reembedder] warning: could not save snapshot: {e}")
    finally:
        session.close()


def reembed(
    new_chunks: list[Document],
    source: str,
    old_chunk_ids: list[str] = None,
    namespace: str = None,
    event_id: int = None,
) -> dict:
    """
    SAFELY replaces specific chunks in Pinecone with snapshot for rollback.

    If old_chunk_ids are provided, only those vectors are deleted.
    If not, NO deletion happens — new chunks are added alongside existing ones.
    This prevents accidental wipeout of the entire source document.

    If event_id is provided, saves a snapshot before deletion (enables rollback).

    Args:
        new_chunks: New rechunked Documents to insert.
        source: Source document identifier (for metadata).
        old_chunk_ids: Specific Pinecone vector IDs to delete (from the failing query).
        namespace: Pinecone namespace.
        event_id: Repair event ID for snapshot tracking.

    Returns: {"old_count": int, "new_count": int, "new_chunk_ids": list}
    """
    ns = namespace or settings.pinecone_namespace
    vs = get_vector_store(ns)
    index = get_pinecone_index()

    old_count = 0

    # Only delete specific chunk IDs — never delete by source filter
    if old_chunk_ids:
        # SNAPSHOT: Save old chunks before deletion (if event_id provided)
        if event_id:
            _save_chunk_snapshot(old_chunk_ids, ns, event_id)

        old_count = len(old_chunk_ids)
        for i in range(0, len(old_chunk_ids), 1000):
            batch = old_chunk_ids[i:i + 1000]
            index.delete(ids=batch, namespace=ns)
        print(f"[reembedder] deleted {old_count} specific vectors")
    else:
        print("[reembedder] no old IDs provided — adding new chunks without deletion")

    # Insert new chunks and capture their IDs
    new_ids = vs.add_documents(new_chunks)

    return {
        "old_count": old_count,
        "new_count": len(new_chunks),
        "new_chunk_ids": new_ids or [],
    }


def rollback_from_snapshot(event_id: int, namespace: str = None,
                           new_chunk_ids: list = None) -> dict:
    """
    Restores old chunks from ChunkSnapshot and deletes the new ones
    that were inserted during the failed repair.

    Process:
      1. DELETE the new chunks that were inserted during the repair
      2. Load snapshot entries for this event_id
      3. Re-embed the original text and upsert back to Pinecone

    Args:
        event_id: The repair event whose chunks should be rolled back.
        namespace: Pinecone namespace.
        new_chunk_ids: IDs of the new chunks to delete (from reembed return).

    Returns: {"restored": int, "failed": int}
    """
    from db.session import get_session
    from db.models import ChunkSnapshot

    ns = namespace or settings.pinecone_namespace
    index = get_pinecone_index()
    embeddings = get_embeddings()
    session = get_session()

    try:
        # ── Step 1: Delete the NEW chunks that were added during repair ──
        if new_chunk_ids:
            for i in range(0, len(new_chunk_ids), 1000):
                batch = new_chunk_ids[i:i + 1000]
                index.delete(ids=batch, namespace=ns)
            print(f"[reembedder] deleted {len(new_chunk_ids)} new chunks from failed repair")

        # ── Step 2: Restore old chunks from snapshot ──
        snapshots = session.query(ChunkSnapshot).filter(
            ChunkSnapshot.event_id == event_id,
            ChunkSnapshot.namespace == ns,
        ).all()

        if not snapshots:
            print(f"[reembedder] no snapshot found for event {event_id} — cannot rollback")
            return {"restored": 0, "failed": 0}

        restored = 0
        failed = 0

        # Re-embed and upsert original chunks back to Pinecone
        upsert_batch = []
        for snap in snapshots:
            if not snap.text:
                failed += 1
                continue
            try:
                fresh_emb = embeddings.embed_query(snap.text)
                metadata = json.loads(snap.metadata_json) if snap.metadata_json else {}
                upsert_batch.append({
                    "id": snap.vector_id,
                    "values": fresh_emb,
                    "metadata": metadata,
                })
                restored += 1
            except Exception as e:
                print(f"[reembedder] rollback failed for {snap.vector_id}: {e}")
                failed += 1

        # Upsert in batches
        for i in range(0, len(upsert_batch), 50):
            batch = upsert_batch[i:i + 50]
            index.upsert(vectors=batch, namespace=ns)

        # Clean up snapshot entries
        session.query(ChunkSnapshot).filter(
            ChunkSnapshot.event_id == event_id,
        ).delete()
        session.commit()

        print(f"[reembedder] rollback complete: restored={restored}, failed={failed}")
        return {"restored": restored, "failed": failed}

    except Exception as e:
        session.rollback()
        print(f"[reembedder] rollback error: {e}")
        return {"restored": 0, "failed": 0}
    finally:
        session.close()

