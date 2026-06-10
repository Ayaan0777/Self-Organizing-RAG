"""
Auto-Chunker Pipeline — SH2: Recursive Only
=============================================
Simplified single-strategy chunker using RecursiveCharacterTextSplitter.

Removed semantic segmentation and clustering strategies — we found
recursive chunking gives the best results for our dataset.

Usage:
    from auto_chunker import auto_chunk

    chunks = auto_chunk(text, source="doc.pdf")
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Use same parameters as ingestion for consistency
CHUNK_SIZE    = 1250
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 200
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def _enforce_min_size(chunks: list[Document], min_chars: int = MIN_CHUNK_SIZE) -> list[Document]:
    """Merges chunks smaller than min_chars with their neighbor."""
    if len(chunks) <= 1:
        return chunks

    result = []
    buffer = chunks[0]

    for chunk in chunks[1:]:
        if len(buffer.page_content) < min_chars:
            buffer = Document(
                page_content=buffer.page_content + " " + chunk.page_content,
                metadata={**buffer.metadata, **chunk.metadata},
            )
        else:
            result.append(buffer)
            buffer = chunk

    if result and len(buffer.page_content) < min_chars:
        last = result.pop()
        buffer = Document(
            page_content=last.page_content + " " + buffer.page_content,
            metadata={**last.metadata, **buffer.metadata},
        )
    result.append(buffer)
    return result


def auto_chunk(
    text: str,
    source: str,
    **kwargs,  # accepts but ignores legacy params like strategy
) -> list[Document]:
    """
    Auto-chunks text using RecursiveCharacterTextSplitter.

    Args:
        text:   Raw document text to chunk.
        source: Source identifier for metadata.

    Returns:
        List of LangChain Documents with chunk sizes between
        MIN_CHUNK_SIZE (200) and CHUNK_SIZE (1250) chars.
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

    # Enforce minimum chunk size
    chunks = _enforce_min_size(chunks, min_chars=MIN_CHUNK_SIZE)

    # Stats
    sizes = [len(c.page_content) for c in chunks]
    print(
        f"[auto-chunker] recursive | "
        f"{len(chunks)} chunks | "
        f"sizes: {min(sizes)}-{max(sizes)} chars"
    )

    return chunks
