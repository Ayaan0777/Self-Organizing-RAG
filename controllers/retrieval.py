from services.llm_factory import get_vector_store, get_llm
from langchain_core.prompts import ChatPromptTemplate
from services.query_logger import log_interaction

def answer_query(query: str, namespace: str = "default", skip_log: bool = False):
    vector_store = get_vector_store(namespace=namespace)
    
    raw_results = vector_store.similarity_search_with_score(query, k=3)
    
    contexts = []
    scores = []
    for doc, score in raw_results:
        contexts.append(doc.page_content)
        scores.append(float(score))

    context_str = "\n\n---\n\n".join(contexts)
    
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the context provided:\n\n{context}\n\nQuestion: {input}"
    )
    
    llm = get_llm()
    chain = prompt | llm
    
    response = chain.invoke({"context": context_str, "input": query})
    answer = response.content

    # Only log if we are NOT running an evaluation
    if not skip_log:
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