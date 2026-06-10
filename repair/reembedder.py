"""
Repair Reembedder — SH2: With Backup + Rollback
=================================================
Safely replaces specific chunks in Pinecone with rollback support.

Flow (used by orchestrator):
  1. backup_chunks()  → fetch old vectors + metadata into memory
  2. delete_chunks()  → remove old vectors from Pinecone
  3. insert_chunks()  → add new rechunked vectors
  4. probe_score()    → re-run query to check improvement
  5a. If improved    → done (new chunks stay, backup discarded)
  5b. If NOT improved → rollback() restores old vectors from backup

This ensures old data is NEVER permanently lost during repair.
"""
import logging
from langchain_core.documents import Document
from services.llm_factory import get_vector_store, get_pinecone_index, get_embeddings
from config import settings


def backup_chunks(
    chunk_ids: list[str],
    namespace: str = None,
) -> dict:
    """
    Fetches old vectors + metadata from Pinecone for backup.
    Must be called BEFORE deleting anything.

    Args:
        chunk_ids: Vector IDs to backup.
        namespace: Pinecone namespace.

    Returns:
        dict: Raw Pinecone fetch response containing {id, values, metadata}
              for each backed-up vector. Pass this to rollback() if needed.
    """
    if not chunk_ids:
        return {}

    ns = namespace or settings.pinecone_namespace
    index = get_pinecone_index()

    # Fetch in batches (Pinecone fetch limit is 1000 IDs)
    all_vectors = {}
    for i in range(0, len(chunk_ids), 1000):
        batch = chunk_ids[i:i + 1000]
        response = index.fetch(ids=batch, namespace=ns)
        if response and hasattr(response, 'vectors'):
            all_vectors.update(response.vectors)

    logging.info(f"[reembedder] backed up {len(all_vectors)} vectors")
    return all_vectors


def delete_chunks(
    chunk_ids: list[str],
    namespace: str = None,
) -> int:
    """
    Deletes specific vectors from Pinecone by ID.

    Args:
        chunk_ids: Vector IDs to delete.
        namespace: Pinecone namespace.

    Returns:
        Number of vectors deleted.
    """
    if not chunk_ids:
        return 0

    ns = namespace or settings.pinecone_namespace
    index = get_pinecone_index()

    for i in range(0, len(chunk_ids), 1000):
        batch = chunk_ids[i:i + 1000]
        index.delete(ids=batch, namespace=ns)

    logging.info(f"[reembedder] deleted {len(chunk_ids)} vectors")
    return len(chunk_ids)


def insert_chunks(
    new_chunks: list[Document],
    namespace: str = None,
) -> list[str]:
    """
    Inserts new rechunked vectors into Pinecone.

    Args:
        new_chunks: New LangChain Documents to embed and insert.
        namespace: Pinecone namespace.

    Returns:
        List of new vector IDs that were inserted.
    """
    if not new_chunks:
        return []

    ns = namespace or settings.pinecone_namespace
    vs = get_vector_store(ns)

    new_ids = vs.add_documents(new_chunks)
    logging.info(f"[reembedder] inserted {len(new_ids)} new vectors")
    return new_ids


def rollback(
    backup_data: dict,
    new_chunk_ids: list[str],
    namespace: str = None,
) -> bool:
    """
    Rollback: removes the new (failed) chunks and restores the old ones.

    Args:
        backup_data:   The dict returned by backup_chunks() — contains
                       old vector data {id: {values, metadata}}.
        new_chunk_ids: IDs of the new chunks that were inserted (to delete).
        namespace:     Pinecone namespace.

    Returns:
        True if rollback succeeded, False on error.
    """
    ns = namespace or settings.pinecone_namespace
    index = get_pinecone_index()

    try:
        # 1. Delete the new (failed) chunks
        if new_chunk_ids:
            for i in range(0, len(new_chunk_ids), 1000):
                batch = new_chunk_ids[i:i + 1000]
                index.delete(ids=batch, namespace=ns)
            logging.info(f"[reembedder] rollback — deleted {len(new_chunk_ids)} new vectors")

        # 2. Restore old chunks from backup
        if backup_data:
            vectors_to_upsert = []
            for vec_id, vec_data in backup_data.items():
                vectors_to_upsert.append({
                    "id": vec_id,
                    "values": vec_data.values if hasattr(vec_data, 'values') else vec_data.get('values', []),
                    "metadata": vec_data.metadata if hasattr(vec_data, 'metadata') else vec_data.get('metadata', {}),
                })

            # Upsert in batches
            for i in range(0, len(vectors_to_upsert), 100):
                batch = vectors_to_upsert[i:i + 100]
                index.upsert(vectors=batch, namespace=ns)

            logging.info(f"[reembedder] rollback — restored {len(vectors_to_upsert)} old vectors")

        return True

    except Exception as e:
        logging.error(f"[reembedder] rollback FAILED: {e}")
        return False


def probe_score(query: str, namespace: str = None) -> float:
    """
    Re-runs a query against Pinecone and returns the top-1 similarity score.
    Used to check if repair improved retrieval quality.

    Args:
        query: The original failing query text.
        namespace: Pinecone namespace.

    Returns:
        Top-1 similarity score (0.0 to 1.0), or 0.0 on error.
    """
    try:
        ns = namespace or settings.pinecone_namespace
        vs = get_vector_store(ns)

        results = vs.similarity_search_with_score(query, k=5)
        if results:
            # LangChain Pinecone returns (doc, score) tuples
            scores = [score for _, score in results]
            return max(scores) if scores else 0.0
        return 0.0
    except Exception as e:
        logging.error(f"[reembedder] probe failed: {e}")
        return 0.0
