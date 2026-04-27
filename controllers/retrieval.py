from services.llm_factory import get_vector_store, get_llm
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from services.query_logger import log_interaction

def answer_query(query: str, namespace: str = "default"):
    vector_store = get_vector_store(namespace=namespace)
    
    # 1. Manual retrieval to capture SCORES (Crucial for Auto-RAG detection)
    raw_results = vector_store.similarity_search_with_score(query, k=3)
    
    contexts = []
    scores = []
    for doc, score in raw_results:
        contexts.append(doc.page_content)
        scores.append(float(score)) # Ensure JSON serializable

    # 2. Format context for the LLM
    context_str = "\n\n---\n\n".join(contexts)
    
    # 3. Generate Answer
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the context provided:\n\n{context}\n\nQuestion: {input}"
    )
    
    llm = get_llm()
    chain = prompt | llm
    
    response = chain.invoke({"context": context_str, "input": query})
    answer = response.content

    # 4. Fire off the logger
    log_interaction(
        namespace=namespace,
        query=query,
        answer=answer,
        contexts=contexts,
        scores=scores
    )
    
    return {
        "answer": answer,
        "retrieved_contexts": contexts,
        "scores": scores
    }