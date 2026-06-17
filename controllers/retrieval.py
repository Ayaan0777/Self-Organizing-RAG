import time
import logging
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
    Full RAG pipeline with retrieval gating:
      0. Retrieval gating  — skip Pinecone for greetings/chitchat
      1. Retrieval         — fetch top-k chunks from Pinecone
      2. Answer generation — single-pass through the stuff chain
      3. Logging + detection
    Uses dynamic K if Strategy 1 has been promoted, otherwise K=5.
    """
    t0 = time.time()

    # ── Step 0: Retrieval Gating ────────────────────────────────
    try:
        from organiser.retrieval_gate import check_retrieval_needed
        gate_result = check_retrieval_needed(query)
    except Exception as e:
        logging.warning(f"[retrieval] gate import/call failed ({e}), defaulting to retrieve")
        gate_result = {"needs_retrieval": True, "reason": "fallback"}

    if not gate_result["needs_retrieval"]:
        return _direct_answer(query, t0, gate_result)

    # ── Step 1: Retrieval ───────────────────────────────────────
    vector_store = get_vector_store(namespace)
    k = _resolve_main_k(query)

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


def _direct_answer(query: str, t0: float, gate_result: dict) -> dict:
    """
    Handles queries that don't need retrieval (greetings, chitchat, etc.).
    Answers directly from the LLM without touching Pinecone.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm()
    direct_response = llm.invoke([
        SystemMessage(content=(
            "You are a document Q&A assistant for a RAG (Retrieval-Augmented Generation) system. "
            "Your ONLY purpose is to help users ask questions about their ingested documents. "
            "When users greet you or send casual messages, respond warmly but ALWAYS guide them "
            "toward asking about their documents. "
            "For example, if they say 'hi', respond like: "
            "'Hello! What would you like to know about your documents?' "
            "or 'Hi there! Feel free to ask any question about the documents you have ingested.' "
            "NEVER suggest activities outside document Q&A (no music, alarms, browsing, etc). "
            "NEVER ask generic questions like 'what is on your mind?' or 'how can I help you today?'. "
            "ALWAYS mention documents, RAG, or ingested data in your response to keep context clear. "
            "Keep responses concise — one or two sentences max."
        )),
        HumanMessage(content=query),
    ]).content
    latency_ms = int((time.time() - t0) * 1000)

    # Still log for tracking — but with empty scores/chunks
    log_id = log_query(
        query           = query,
        scores          = [],
        answer          = direct_response,
        latency_ms      = latency_ms,
        chunk_metadatas = [],
        chunk_contents  = [],
    )
    # Skip run_detectors — retrieval-based detection is irrelevant for direct answers

    return {
        "answer"            : direct_response,
        "retrieved_contexts": [],
        "scores"            : [],
        "log_id"            : log_id,
        "gate"              : "direct",
        "gate_detail"       : gate_result.get("gate_detail", ""),
    }
