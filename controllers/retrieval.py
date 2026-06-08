import time
from services.llm_factory import get_vector_store, get_llm
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from logger.query_logger import log_query
from detector.detectors import run_detectors


def answer_query(query: str, namespace: str = None):
    """
    Full RAG pipeline: retrieve from Pinecone, generate answer, log, detect.
    """
    vector_store = get_vector_store(namespace)

    t0 = time.time()
    docs_with_scores = vector_store.similarity_search_with_score(query, k=5)
    docs   = [d for d, _ in docs_with_scores]
    # Pinecone returns cosine similarity directly (1 = identical, 0 = unrelated)
    scores = [round(float(s), 4) for _, s in docs_with_scores]

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the context provided:\n\n{context}\n\nQuestion: {input}"
    )
    llm = get_llm()
    document_chain  = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(search_kwargs={"k": 5}), document_chain
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


def generate_answer_only(query: str, namespace: str = None):
    """
    Full RAG answer generation without logging or detector side effects.
    """
    vector_store = get_vector_store(namespace)

    docs_with_scores = vector_store.similarity_search_with_score(query, k=5)
    scores = [round(float(s), 4) for _, s in docs_with_scores]

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the context provided:\n\n{context}\n\nQuestion: {input}"
    )
    llm = get_llm()
    document_chain  = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        vector_store.as_retriever(search_kwargs={"k": 5}), document_chain
    )
    response = retrieval_chain.invoke({"input": query})

    return {
        "answer"            : response["answer"],
        "retrieved_contexts": [doc.page_content for doc in response["context"]],
        "scores"            : scores,
    }
