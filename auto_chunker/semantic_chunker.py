"""
Semantic Segmentation Chunker
=============================
Splits text into chunks at natural topic boundaries detected via
embedding similarity between consecutive sentences.

Algorithm:
  1. Split text into sentences
  2. Embed each sentence using the configured embedding model
  3. Compute cosine similarity between consecutive sentence embeddings
  4. Where similarity drops below a threshold → topic boundary
  5. Group sentences between boundaries into chunks
"""
import re
import numpy as np
from langchain_core.documents import Document


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex (handles ., ?, !)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def chunk_by_semantic_segmentation(
    text: str,
    source: str,
    similarity_threshold: float = 0.65,
) -> list[Document]:
    """
    Splits text at points where consecutive sentences have low embedding similarity.
    These are natural topic transition points.

    Args:
        text: Full document text.
        source: Source metadata for the chunks.
        similarity_threshold: Similarity below this = topic boundary.

    Returns:
        List of LangChain Documents, one per detected segment.
    """
    from services.llm_factory import get_embeddings

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [Document(page_content=text, metadata={"source": source, "strategy": "semantic_segmentation"})]

    emb_model = get_embeddings()
    embeddings = [np.array(emb_model.embed_query(s[:500])) for s in sentences]

    # Find topic boundaries: where consecutive similarity drops
    boundaries = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_sim(embeddings[i], embeddings[i + 1])
        if sim < similarity_threshold:
            boundaries.append(i + 1)  # boundary AFTER sentence i

    # Build chunks from boundary indices
    chunks = []
    prev = 0
    for b in boundaries:
        segment = " ".join(sentences[prev:b])
        if segment.strip():
            chunks.append(Document(
                page_content=segment,
                metadata={"source": source, "strategy": "semantic_segmentation"},
            ))
        prev = b

    # Last segment
    segment = " ".join(sentences[prev:])
    if segment.strip():
        chunks.append(Document(
            page_content=segment,
            metadata={"source": source, "strategy": "semantic_segmentation"},
        ))

    return chunks if len(chunks) > 1 else [
        Document(page_content=text, metadata={"source": source, "strategy": "semantic_segmentation"})
    ]
