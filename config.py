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


    # Stage 2-4: Self-healing thresholds
    precision_threshold: float = 0.4       # retrieval precision below this triggers repair (decision engine check)
    sufficiency_threshold: float = 0.7     # context sufficiency check threshold
    hallucination_threshold: float = 0.3   # hallucination rate above this triggers repair (decision engine check)

    # Auto-Worker Trigger settings
    pending_min: int = 5
    pending_ratio: float = 0.30
    poll_interval_seconds: int = 5

    # Detector settings (Low Recall Trigger thresholds)
    score_low: float = 0.15
    score_drop: float = 0.15
    coherence_ratio: float = 0.65
    evidence_match: float = 0.60

    # Stage 2 Quality Metrics thresholds
    precision_relevance_threshold: float = 0.50
    hallucination_grounding_threshold: float = 0.55

    # Repair & Promotion thresholds
    score_cliff_threshold: float = 0.12
    promotion_threshold: int = 5

    # Ground-truth dataset for the gt_lookup module. Path is relative to the
    # project root. Edit here (or in .env) to switch datasets without code changes.
    gt_dataset_path: str = "dataset/long_ans.json"


    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
