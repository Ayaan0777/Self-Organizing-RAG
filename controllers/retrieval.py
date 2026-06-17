import time
from services.llm_factory import get_vector_store, get_llm
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from logger.query_logger import log_query
from detector.detectors import run_detectors


def _resolve_main_k(query: str) -> int:
    """
    Determines the retrieval K for the main query pipeline.

    If Strategy 1 (dynamic K) has been promoted to the main pipeline,
    classifies the query and returns a dynamic K. Otherwise returns 5.

    Promotion is a one-way flag set by repair/cascade.py when S1
    accumulates >= 5 successes.
    """
    try:
        from db.session import get_session
        from db.models import RuntimeFlag

        session = get_session()
        try:
            promoted = (
                session.query(RuntimeFlag)
                .filter(RuntimeFlag.name == "dynamic_k_promoted",
                        RuntimeFlag.value == True)
                .first()
            )
        finally:
            session.close()

        if not promoted:
            return 5

        # Promoted path — classify and pick dynamic K
        from controllers.metrics import classify_question
        from repair.orchestrator import _dynamic_k_selection
        category = classify_question(query)
        k = _dynamic_k_selection(query, category, scores=None)
        return k

    except Exception:
        return 5  # safe fallback


def answer_query(query: str, namespace: str = None):
    """
    Full RAG pipeline: retrieve from Pinecone, generate answer, log, detect.
    Uses dynamic K if Strategy 1 has been promoted, otherwise K=5.
    """
    vector_store = get_vector_store(namespace)
    k = _resolve_main_k(query)

    t0 = time.time()
    docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
    docs   = [d for d, _ in docs_with_scores]
    # Pinecone returns cosine similarity directly (1 = identical, 0 = unrelated)
    scores = [round(float(s), 4) for _, s in docs_with_scores]

    prompt = ChatPromptTemplate.from_template(
        "You are a precise factual assistant. Answer the question using ONLY the context below. "
        "Extract the answer directly from the text. Do NOT use any outside knowledge. "
        "If the exact answer appears in the context, state it clearly and concisely.\n\n"
        "Context:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    )
    llm = get_llm()
    document_chain  = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(search_kwargs={"k": k}), document_chain
    )
    response   = retrieval_chain.invoke({"input": query})
    latency_ms = int((time.time() - t0) * 1000)

    # Persist query to DB then run detection rules
    log_id = log_query(
        query           = query,
        scores          = scores,
        answer          = response["answer"],
        latency_ms      = latency_ms,
        chunk_metadatas = [d.metadata for d in docs],
        chunk_contents  = [d.page_content for d in docs],
    )
    run_detectors(log_id)   # fires silently, never blocks the response

    return {
        "answer"            : response["answer"],
        "retrieved_contexts": [doc.page_content for doc in response["context"]],
        "scores"            : scores,
        "log_id"            : log_id,
    }


def generate_answer_only(query: str, namespace: str = None, k: int = None):
    """
    Full RAG answer generation without logging or detector side effects.
    If k is provided, uses it directly (callers like _probe_metrics need
    answer + chunks to come from the SAME K). Otherwise defers to
    _resolve_main_k: dynamic K if Strategy 1 has been promoted, else 5.
    """
    vector_store = get_vector_store(namespace)
    if k is None:
        k = _resolve_main_k(query)

    docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
    scores = [round(float(s), 4) for _, s in docs_with_scores]

    prompt = ChatPromptTemplate.from_template(
        "You are a precise factual assistant. Answer the question using ONLY the context below. "
        "Extract the answer directly from the text. Do NOT use any outside knowledge. "
        "If the exact answer appears in the context, state it clearly and concisely.\n\n"
        "Context:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    )
    llm = get_llm()
    document_chain  = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(search_kwargs={"k": k}), document_chain
    )
    response = retrieval_chain.invoke({"input": query})

    return {
        "answer"            : response["answer"],
        "retrieved_contexts": [doc.page_content for doc in response["context"]],
        "scores"            : scores,
    }

