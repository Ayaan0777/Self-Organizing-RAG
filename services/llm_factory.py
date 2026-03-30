from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import settings


# Initialize Pinecone client
pc = Pinecone(api_key=settings.pinecone_api_key)


def get_embeddings():
    """
    Returns the embedding model.
    Currently using local Ollama embeddings.
    """
    return OllamaEmbeddings(
        model=settings.embedding_model_name,
        base_url=settings.ollama_base_url
    )


def get_vector_store(namespace: str = "default", index_name: str = None):
    """
    Connects to Pinecone vector store using the embedding model.
    Optionally override the index name (e.g. rag-index-768, rag-index-1024).
    Falls back to PINECONE_INDEX_NAME from .env if not specified.
    """
    target_index = index_name or settings.pinecone_index_name
    return PineconeVectorStore(
        index_name=target_index,
        embedding=get_embeddings(),
        pinecone_api_key=settings.pinecone_api_key,
        namespace=namespace
    )


def get_llm():
    """
    Returns the configured LLM depending on provider.
    Supported providers:
        - ollama
        - gemini
    """

    if settings.llm_provider.lower() == "ollama":
        return ChatOllama(
            model=settings.llm_model_name,
            base_url=settings.ollama_base_url,
            temperature=0.2
        )

    elif settings.llm_provider.lower() == "gemini":
        return ChatGoogleGenerativeAI(
            model=settings.llm_model_name,
            google_api_key=settings.gemini_api_key,
            temperature=0.2
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: {settings.llm_provider}"
        )