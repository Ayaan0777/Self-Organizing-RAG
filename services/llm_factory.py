from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings


# Initialize Pinecone client
pc = Pinecone(api_key=settings.pinecone_api_key)


def get_embeddings():
    """
    Returns the embedding model dynamically based on the .env provider.
    """
    provider = getattr(settings, "embedding_provider", "ollama").lower()

    if provider == "huggingface":
        # BGE models require normalization for Cosine Similarity
        encode_kwargs = {'normalize_embeddings': True} 
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={'device': 'cpu'}, # Change to 'cuda' if you have an Nvidia GPU
            encode_kwargs=encode_kwargs
        )
    else:
        # Default back to Ollama
        return OllamaEmbeddings(
            model=settings.embedding_model_name,
            base_url=settings.ollama_base_url
        )


def get_vector_store(namespace: str = "default"):  # <-- 1. Add namespace parameter
    """
    Connects to Pinecone vector store using the embedding model.
    """
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=get_embeddings(),
        pinecone_api_key=settings.pinecone_api_key,
        namespace=namespace  # <-- 2. Pass it to Pinecone
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