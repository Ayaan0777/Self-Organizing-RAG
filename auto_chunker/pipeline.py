"""
Auto-Chunker Pipeline — Month 3 Deliverable
=============================================
Orchestrates the full auto-chunking pipeline:

  text → semantic segmentation OR clustering → adaptive sizing → final chunks

Usage:
    from auto_chunker import auto_chunk

    chunks = auto_chunk(text, source="doc.pdf", strategy="semantic")
    chunks = auto_chunk(text, source="doc.pdf", strategy="clustering")
"""
from langchain_core.documents import Document

from auto_chunker.semantic_chunker import chunk_by_semantic_segmentation
from auto_chunker.cluster_chunker import chunk_by_clustering
from auto_chunker.adaptive_chunker import adaptive_resize


def auto_chunk(
    text: str,
    source: str,
    strategy: str = "semantic",
    similarity_threshold: float = 0.65,
    distance_threshold: float = 0.5,
) -> list[Document]:
    """
    Full auto-chunking pipeline: segment → adaptive resize → final chunks.

    Args:
        text: Raw document text.
        source: Source identifier for metadata.
        strategy: "semantic" (embedding boundary detection) or
                  "clustering" (agglomerative sentence clustering).
        similarity_threshold: For semantic strategy — cosine sim below this = boundary.
        distance_threshold: For clustering strategy — cosine distance for cluster merge.

    Returns:
        List of LangChain Documents with adaptive sizing (200–2000 tokens each).
    """
    if not text or not text.strip():
        return []

    # Step 1: Initial segmentation
    if strategy == "clustering":
        raw_chunks = chunk_by_clustering(
            text, source,
            distance_threshold=distance_threshold,
        )
    else:
        raw_chunks = chunk_by_semantic_segmentation(
            text, source,
            similarity_threshold=similarity_threshold,
        )

    # Step 2: Adaptive sizing (merge small, split large)
    final_chunks = adaptive_resize(raw_chunks)

    # Stats
    sizes = [len(c.page_content) for c in final_chunks]
    print(f"[auto-chunker] strategy={strategy} | "
          f"raw={len(raw_chunks)} → final={len(final_chunks)} chunks | "
          f"sizes: {min(sizes)}-{max(sizes)} chars")

    return final_chunks
