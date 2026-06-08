"""
Sentence Similarity Clustering Chunker
=======================================
Groups semantically similar sentences into clusters using
agglomerative clustering on sentence embeddings, then reassembles
each cluster in document order to form coherent chunks.

Algorithm:
  1. Split text into sentences
  2. Embed all sentences
  3. Compute pairwise cosine distance matrix
  4. Agglomerative clustering with distance threshold
  5. For each cluster, collect sentences in original order → one chunk
"""
import re
import numpy as np
from langchain_core.documents import Document


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def chunk_by_clustering(
    text: str,
    source: str,
    distance_threshold: float = 0.5,
    max_clusters: int = 20,
) -> list[Document]:
    """
    Groups sentences into semantic clusters via agglomerative clustering.
    Sentences within each cluster are kept in original document order.

    Args:
        text: Full document text.
        source: Source metadata.
        distance_threshold: Cosine distance threshold for cluster merging.
        max_clusters: Maximum number of clusters to produce.

    Returns:
        List of LangChain Documents, one per cluster.
    """
    from services.llm_factory import get_embeddings
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_distances

    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        return [Document(page_content=text, metadata={"source": source, "strategy": "clustering"})]

    emb_model = get_embeddings()
    embeddings = np.array([emb_model.embed_query(s[:500]) for s in sentences])

    # Cosine distance matrix
    dist_matrix = cosine_distances(embeddings)

    # Agglomerative clustering
    n_clusters = min(max_clusters, len(sentences) // 2) or 1
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(dist_matrix)

    # Group sentences by cluster label, preserving document order
    cluster_map: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        cluster_map.setdefault(label, []).append(idx)

    # Sort clusters by the earliest sentence index to maintain reading order
    sorted_clusters = sorted(cluster_map.values(), key=lambda idxs: idxs[0])

    chunks = []
    for sentence_idxs in sorted_clusters:
        content = " ".join(sentences[i] for i in sentence_idxs)
        if content.strip():
            chunks.append(Document(
                page_content=content,
                metadata={"source": source, "strategy": "clustering"},
            ))

    return chunks if len(chunks) > 1 else [
        Document(page_content=text, metadata={"source": source, "strategy": "clustering"})
    ]
