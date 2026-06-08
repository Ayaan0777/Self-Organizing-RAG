from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import settings

# ── Cached singleton instances ──
_cached_embeddings = None
_cached_llm = None
_cached_pinecone_index = None


def get_embeddings():
    """
    Returns the Ollama embedding model (cached after first call).
    Default: mxbai-embed-large (1024 dims).
    """
    global _cached_embeddings
    if _cached_embeddings is not None:
        return _cached_embeddings

    _cached_embeddings = OllamaEmbeddings(
        model=settings.embedding_model_name,
        base_url=settings.ollama_base_url,
    )
    return _cached_embeddings


def get_pinecone_index():
    """Returns a cached Pinecone Index object."""
    global _cached_pinecone_index
    if _cached_pinecone_index is not None:
        return _cached_pinecone_index

    pc = Pinecone(api_key=settings.pinecone_api_key)
    _cached_pinecone_index = pc.Index(settings.pinecone_index_name)
    return _cached_pinecone_index


def get_vector_store(namespace: str = None):
    """
    Returns a LangChain PineconeVectorStore for the given namespace.
    Falls back to the default namespace from .env if none provided.
    """
    ns = namespace or settings.pinecone_namespace
    return PineconeVectorStore(
        index=get_pinecone_index(),
        embedding=get_embeddings(),
        namespace=ns,
    )


def get_llm():
    """
    Returns the configured LLM (cached after first call).
    Default: Ollama mistral.
    """
    global _cached_llm
    if _cached_llm is not None:
        return _cached_llm

    if settings.llm_provider.lower() == "ollama":
        _cached_llm = ChatOllama(
            model=settings.llm_model_name,
            base_url=settings.ollama_base_url,
            temperature=0.2,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
    return _cached_llm