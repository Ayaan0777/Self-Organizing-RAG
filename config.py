from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):

    pinecone_api_key: str
    pinecone_index_name: str
    embedding_provider: str="ollama"  
    embedding_model_name: str
    ollama_base_url: str
    llm_model_name: str

    llm_provider: str = "ollama"   # ← add this

    gemini_api_key: Optional[str] = None

    class Config:
        env_file = ".env"

settings = Settings()