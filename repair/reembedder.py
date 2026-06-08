"""
Repair Reembedder — SAFE partial re-embedding
==============================================
Only replaces the specific chunks related to a failing query.
NEVER deletes all vectors for a source — that would wipe the entire document.

Strategy:
  1. Find the specific vector IDs of chunks retrieved for the failing query
  2. Delete ONLY those specific IDs
  3. Insert new rechunked versions alongside the existing index
"""
from services.llm_factory import get_vector_store, get_pinecone_index, get_embeddings
from langchain_core.documents import Document
from config import settings


def reembed(
    new_chunks: list[Document],
    source: str,
    old_chunk_ids: list[str] = None,
    namespace: str = None,
) -> dict:
    """
    SAFELY replaces specific chunks in Pinecone.

    If old_chunk_ids are provided, only those vectors are deleted.
    If not, NO deletion happens — new chunks are added alongside existing ones.
    This prevents accidental wipeout of the entire source document.

    Args:
        new_chunks: New rechunked Documents to insert.
        source: Source document identifier (for metadata).
        old_chunk_ids: Specific Pinecone vector IDs to delete (from the failing query).
        namespace: Pinecone namespace.

    Returns: {"old_count": int, "new_count": int}
    """
    ns = namespace or settings.pinecone_namespace
    vs = get_vector_store(ns)
    index = get_pinecone_index()

    old_count = 0

    # Only delete specific chunk IDs — never delete by source filter
    if old_chunk_ids:
        old_count = len(old_chunk_ids)
        for i in range(0, len(old_chunk_ids), 1000):
            batch = old_chunk_ids[i:i + 1000]
            index.delete(ids=batch, namespace=ns)
        print(f"[reembedder] deleted {old_count} specific vectors")
    else:
        print("[reembedder] no old IDs provided — adding new chunks without deletion")

    # Insert new chunks
    vs.add_documents(new_chunks)

    return {"old_count": old_count, "new_count": len(new_chunks)}
