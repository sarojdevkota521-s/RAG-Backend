from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    APP_NAME: str = "Conversational RAG API"
    DEBUG: bool = True

    # SQLite Database Configuration
    DATABASE_URL: str = "sqlite:///./rag_backend.db"

    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PROTOCOL: int = 2
    CHAT_MEMORY_TTL: int = 86400  # 24 hours in seconds

    # Qdrant Vector Store Configuration
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    COLLECTION_NAME: str = "documents"
    QDRANT_PREFER_GRPC: bool = False

    # Embedding Model Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # LLM Provider Configuration
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_MODEL: str = "meta-llama/llama-3.1-8b-instruct"

    # Additional API Keys (supported as fallbacks)
    GROQ_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def clean_key(self, key: Optional[str]) -> Optional[str]:
        """Strip enclosing quotes and whitespace from environment strings."""
        if not key:
            return None
        cleaned = key.strip().strip("'\"")
        return cleaned if cleaned and cleaned != "your_api_key" else None

    @property
    def effective_llm_credentials(self) -> tuple[Optional[str], str, str]:
        """
        Determines the effective API key, base URL, and model.
        Supports explicit LLM settings, Groq, xAI, or OpenAI keys.
        """
        cleaned_llm_key = self.clean_key(self.LLM_API_KEY)
        cleaned_groq_key = self.clean_key(self.GROQ_API_KEY)
        cleaned_openai_key = self.clean_key(self.OPENAI_API_KEY)

        if cleaned_llm_key:
            return cleaned_llm_key, self.LLM_BASE_URL.strip().strip("'\""), self.LLM_MODEL.strip().strip("'\"")

        if cleaned_groq_key:
            if cleaned_groq_key.startswith("xai-"):
                return cleaned_groq_key, "https://api.x.ai/v1", "grok-2-latest"
            return cleaned_groq_key, "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"

        if cleaned_openai_key:
            return cleaned_openai_key, "https://api.openai.com/v1", "gpt-4o-mini"

        return None, self.LLM_BASE_URL, self.LLM_MODEL


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()

