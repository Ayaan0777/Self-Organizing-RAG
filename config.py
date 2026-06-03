from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = ""
    pinecone_namespace: str = "mxbai-embed-large"

    # Ollama Embeddings
    embedding_model_name: str = "mxbai-embed-large"
    ollama_base_url: str = "http://localhost:11434"

    # LLM
    llm_provider: str = "ollama"
    llm_model_name: str = "mistral"

    # Gemini (for evaluation)
    gemini_api_key: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
