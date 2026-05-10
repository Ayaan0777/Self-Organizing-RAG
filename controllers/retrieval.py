"""
Retrieval controller — answers queries with top-K scored retrieval,
latency measurement, similarity metrics, and automatic query logging.
"""

import time
import numpy as np
from services.llm_factory import get_vector_store, get_llm, get_embeddings
from langchain_core.prompts import ChatPromptTemplate
from services.query_logger import log_query


def answer_query(
    query: str,
    namespace: str = "default",
    index_name: str = None,
    ground_truth: str = None,
):
    """
    Retrieve relevant chunks, generate an LLM answer, and log diagnostics.

    Args:
        query:        The user question.
        namespace:    Pinecone namespace to search.
        index_name:   Override Pinecone index name.
        ground_truth: Optional ground-truth answer for computing answer similarity.

    Returns:
        dict with answer, retrieved_contexts, scores, latency_ms.
    """
    start = time.perf_counter()

    vector_store = get_vector_store(namespace=namespace, index_name=index_name)

    # ── Top-K retrieval with scores ──────────────────────────────────
    results_with_scores = vector_store.similarity_search_with_score(query, k=5)

    chunks = [doc.page_content for doc, _score in results_with_scores]
    scores = [float(s) for _doc, s in results_with_scores]

    # ── LLM generation ───────────────────────────────────────────────
    context = "\n\n".join(chunks)
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the context provided:\n\n"
        "{context}\n\nQuestion: {input}"
    )
    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({"context": context, "input": query})
    answer = response.content

    elapsed_ms = (time.perf_counter() - start) * 1000

    # ── Similarity metrics ───────────────────────────────────────────
    # ctx_q_sim: use mean of Pinecone cosine-similarity scores (fast, no extra API calls)
    ctx_q_sim = float(np.mean(scores)) if scores else None

    # answer_sem_sim: requires embedding the answer and ground truth
    answer_sem_sim = None
    if ground_truth:
        try:
            embeddings = get_embeddings()
            ans_emb = np.array(embeddings.embed_query(answer[:2000]))
            gt_emb = np.array(embeddings.embed_query(ground_truth[:2000]))
            cos = float(
                np.dot(ans_emb, gt_emb)
                / (np.linalg.norm(ans_emb) * np.linalg.norm(gt_emb))
            )
            answer_sem_sim = cos
        except Exception as e:
            print(f"  WARNING: Answer similarity computation failed: {e}")

    # ── Log to database ──────────────────────────────────────────────
    log_query(
        query=query,
        llm_response=answer,
        top_k_scores=scores,
        retrieved_chunks=chunks,
        latency_ms=elapsed_ms,
        ctx_q_sim=ctx_q_sim,
        answer_sem_sim=answer_sem_sim,
    )

    return {
        "answer": answer,
        "retrieved_contexts": chunks,
        "scores": scores,
        "latency_ms": round(elapsed_ms, 2),
    }