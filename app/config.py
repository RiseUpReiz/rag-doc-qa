from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings, loaded from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "RAG Document Q&A"
    google_api_key: str | None = None

    # Ingestion settings
    data_dir: str = "data"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Embedding and vector store settings
    embedding_provider: str = "google"
    embedding_model: str = "gemini-embedding-001"
    persist_dir: str = "storage"
    collection_name: str = "documents"

    # LLM settings
    llm_provider: str = "google"
    llm_model: str = "gemini-3.6-flash"
    temperature: float = 0.0
    retriever_k: int = 4

    # Eval judge settings (kept separate so the judge can be a different model from llm_model)
    judge_provider: str = "google"
    judge_model: str | None = None


settings = Settings()