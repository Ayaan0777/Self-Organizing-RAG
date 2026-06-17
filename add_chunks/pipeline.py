"""
Add Chunks Pipeline — Recursive Only
======================================
Uses the same RecursiveCharacterTextSplitter and parameters as
controllers/ingestion.py to ensure consistency across all chunking.

  text → recursive split (1250 max, 200 overlap) → min-size enforcement (500) → final chunks

Usage:
    from add_chunks import add_chunk

    chunks = add_chunk(text, source="doc.pdf")
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from controllers.ingestion import (
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE,
    SEPARATORS, _enforce_min_chunk_size,
)


def add_chunk(
    text: str,
    source: str,
    **kwargs,  # accepts but ignores legacy params
) -> list[Document]:
    """
    Chunks text using RecursiveCharacterTextSplitter with the same
    parameters as ingestion (variable size 500–1250 chars).

    Args:
        text:   Raw document text to chunk.
        source: Source identifier for metadata.

    Returns:
        List of LangChain Documents with chunk sizes between
        MIN_CHUNK_SIZE (500) and CHUNK_SIZE (1250) chars.
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )

    chunks = splitter.create_documents(
        [text],
        metadatas=[{"source": source, "strategy": "recursive"}],
    )

    # Enforce minimum chunk size — merge tiny chunks with neighbors
    chunks = _enforce_min_chunk_size(chunks, min_chars=MIN_CHUNK_SIZE)

    # Stats
    sizes = [len(c.page_content) for c in chunks]
    print(
        f"[add-chunks] recursive | "
        f"{len(chunks)} chunks | "
        f"sizes: {min(sizes)}-{max(sizes)} chars"
    )

    return chunks
