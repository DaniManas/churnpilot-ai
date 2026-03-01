from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "ML AI Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Auth ──────────────────────────────────────────────────────────────────
    VALID_API_KEYS: str = "dev-key-123"  # comma-separated in .env

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./data/predictions.db"

    # ── ML Model ──────────────────────────────────────────────────────────────
    MODEL_PATH: str = "ml/artifacts/model.pkl"
    SCALER_PATH: str = "ml/artifacts/scaler.pkl"
    METADATA_PATH: str = "ml/artifacts/model_metadata.json"

    # ── RAG ───────────────────────────────────────────────────────────────────
    EMBEDDING_PROVIDER: str = "local"   # "local" or "openai"
    LLM_PROVIDER: str = "local"         # "local" or "openai"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    FAISS_INDEX_PATH: str = "data/faiss_index"
    DOCS_PATH: str = "data/docs"

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT: str = "20/minute"

    def get_valid_keys(self) -> list[str]:
        """Return API keys as a clean list."""
        return [k.strip() for k in self.VALID_API_KEYS.split(",")]

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance.
    lru_cache means this is created once and reused — not re-read from
    disk on every request. This is the standard FastAPI pattern.
    """
    return Settings()


settings = get_settings()
