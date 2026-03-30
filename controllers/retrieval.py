from services.llm_factory import get_vector_store, get_llm
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def answer_query(query: str, namespace: str = "default"):
    vector_store = get_vector_store(namespace=namespace)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the context provided:\n\n{context}\n\nQuestion: {input}"
    )
    
    llm = get_llm()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({"input": query})
    
    return {
        "answer": response["answer"],
        "retrieved_contexts": [doc.page_content for doc in response["context"]]
    }