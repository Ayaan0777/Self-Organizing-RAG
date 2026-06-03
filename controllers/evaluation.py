from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from controllers.retrieval import answer_query
from services.llm_factory import get_llm, get_embeddings

def calculate_metrics(question: str, ground_truth: str):
    # 1. Get the RAG system's response
    rag_output = answer_query(question)
    
    # 2. Format for Ragas
    data = {
        "question": [question],
        "answer": [rag_output["answer"]],
        "contexts": [rag_output["retrieved_contexts"]],
        "ground_truth": [ground_truth]
    }
    dataset = Dataset.from_dict(data)
    
    # 3. Evaluate, overriding the default OpenAI models with our factory instances
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall],
        llm=get_llm(),
        embeddings=get_embeddings()
    )
    
    return result.to_pandas().to_dict(orient="records")[0]