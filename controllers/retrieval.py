import time
import logging
from services.llm_factory import get_vector_store, get_llm
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from logger.query_logger import log_query
from detector.detectors import run_detectors


def answer_query(
    query: str,
    namespace: str = None,
    k: int = 5,
    rerank: bool = True,
    metadata_filter: dict = None,
):
    """
    Full RAG pipeline with Phase 2 + Phase 4 enhancements:
      0. Retrieval gating  — skip Pinecone for greetings/chitchat
      1. Retrieval         — dynamic k, optional metadata filter
      2. Reranking         — LLM re-scores chunks for better relevance
      3. Query reform.     — rephrase + retry if scores are poor
      4. Answer generation — single-pass through the stuff chain
      5. Logging + detection
    """
    t0 = time.time()

    # ── Step 0: Retrieval Gating (Phase 4) ──────────────────────
    try:
        from organiser.retrieval_gate import check_retrieval_needed
        gate_result = check_retrieval_needed(query)
    except Exception as e:
        logging.warning(f"[retrieval] gate import/call failed ({e}), defaulting to retrieve")
        gate_result = {"needs_retrieval": True, "reason": "fallback"}

    if not gate_result["needs_retrieval"]:
        return _direct_answer(query, t0, gate_result)

    # ── Step 1: Retrieval with over-fetch for dynamic K ───────────
    vector_store = get_vector_store(namespace)

    # Fetch more than requested so dynamic K has room to prune intelligently
    fetch_k = min(k * 2, 15)
    search_kwargs = {"k": fetch_k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    docs_with_scores = vector_store.similarity_search_with_score(query, **search_kwargs)
    docs   = [d for d, _ in docs_with_scores]
    # Pinecone returns cosine similarity directly (1 = identical, 0 = unrelated)
    scores = [round(float(s), 4) for _, s in docs_with_scores]

    # ── Step 2: Reranking (Phase 2) ─────────────────────────────
    if rerank and len(docs) > 1:
        try:
            from services.reranker import rerank_chunks
            docs, scores = rerank_chunks(query, docs, scores)
        except Exception as e:
            logging.warning(f"[retrieval] reranker failed ({e}), keeping original order")

    # ── Step 2.5: Dynamic K Selection (Phase 2) ─────────────────
    try:
        from organiser.dynamic_k import select_optimal_chunks
        docs, scores = select_optimal_chunks(query, docs, scores, target_k=k)
    except Exception as e:
        logging.warning(f"[retrieval] dynamic_k failed ({e}), using first {k} chunks")
        docs = docs[:k]
        scores = scores[:k]

    # ── Step 3: Query Reformulation (Phase 4) ───────────────────
    reformulated_query = None
    if scores and scores[0] < 0.45:
        try:
            from organiser.query_reformulator import try_reformulate
            reform_result = try_reformulate(query, vector_store, k, scores[0])
            if reform_result:
                docs   = reform_result["docs"]
                scores = reform_result["scores"]
                reformulated_query = reform_result["reformulated_query"]
        except Exception as e:
            logging.warning(f"[retrieval] reformulator failed ({e}), keeping original results")


    # ── Step 4: Generate Answer ─────────────────────────────────
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the context provided:\n\n{context}\n\nQuestion: {input}"
    )
    llm = get_llm()
    document_chain = create_stuff_documents_chain(llm, prompt)
    answer_text = document_chain.invoke({"input": query, "context": docs})

    latency_ms = int((time.time() - t0) * 1000)

    # ── Step 5: Persist + Detect ────────────────────────────────
    log_id = log_query(
        query           = query,
        scores          = scores,
        answer          = answer_text,
        latency_ms      = latency_ms,
        chunk_metadatas = [d.metadata for d in docs],
        chunk_contents  = [d.page_content for d in docs],
    )
    run_detectors(log_id)   # fires silently, never blocks the response

    result = {
        "answer"            : answer_text,
        "retrieved_contexts": [doc.page_content for doc in docs],
        "scores"            : scores,
        "log_id"            : log_id,
    }

    # Include extra info when Phase 4 features activated
    if reformulated_query:
        result["reformulated_query"] = reformulated_query
    if not gate_result.get("needs_retrieval", True):
        result["gate"] = "direct"

    return result


def _direct_answer(query: str, t0: float, gate_result: dict) -> dict:
    """
    Handles queries that don't need retrieval (greetings, chitchat, etc.).
    Answers directly from the LLM without touching Pinecone.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm()
    direct_response = llm.invoke([
        SystemMessage(content=(
            "You are a helpful document Q&A assistant. "
            "You answer questions based on ingested documents. "
            "You cannot play music, set alarms, browse the internet, or do anything outside of answering questions. "
            "Keep your responses concise and friendly."
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