"""
Configuration module for RAG application.
Loads settings from environment variables and provides configuration objects.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application Settings
    app_name: str = Field(default="Cost-Efficient RAG", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    
    # Vector Store Settings
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(default="rag_collection", alias="CHROMA_COLLECTION_NAME")
    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        alias="EMBEDDING_MODEL_NAME"
    )
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    
    # Chunking Settings
    default_chunk_size: int = Field(default=1000, alias="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(default=200, alias="DEFAULT_CHUNK_OVERLAP")
    
    # Retrieval Settings
    default_top_k: int = Field(default=5, alias="DEFAULT_TOP_K")
    similarity_threshold: float = Field(default=0.7, alias="SIMILARITY_THRESHOLD")
    
    # LLM Settings
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen2.5:7b", alias="OLLAMA_MODEL")
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    
    # Logging Settings
    log_dir: str = Field(default="./logs", alias="LOG_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_rotation: str = Field(default="500 MB", alias="LOG_ROTATION")
    log_retention: str = Field(default="30 days", alias="LOG_RETENTION")
    
    # Evaluation Settings
    evaluation_dir: str = Field(default="./evaluation", alias="EVALUATION_DIR")
    evaluation_k_values: str = Field(default="3,5,10", alias="EVALUATION_K_VALUES")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"
    
    @property
    def k_values(self) -> list[int]:
        """Parse evaluation K values from string to list of integers."""
        return [int(k.strip()) for k in self.evaluation_k_values.split(",")]


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
