from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = ""
    pinecone_namespace: str = "" 

    # Ollama Embeddings
    embedding_model_name: str = "mxbai-embed-large"
    ollama_base_url: str = "http://localhost:11434"

    # LLM
    llm_provider: str = "ollama"
    llm_model_name: str = "mistral"
    fallback_llm_model: str = "gemma3:27b"

    # Gemini (for evaluation)
    gemini_api_key: str = ""

    # Stage 2-4: Self-healing thresholds
    precision_threshold: float = 0.5       # retrieval precision below this triggers repair
    sufficiency_threshold: float = 0.7     # context sufficiency check threshold
    hallucination_threshold: float = 0.2   # hallucination rate above this triggers repair

    # Ground-truth dataset for the gt_lookup module. Path is relative to the
    # project root. Edit here (or in .env) to switch datasets without code changes.
    gt_dataset_path: str = "dataset/long_ans.json"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
