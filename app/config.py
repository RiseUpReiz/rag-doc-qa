from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

VALID_DEFENCES = ("prompt", "trust", "links")


def parse_defences(value: str | list[str] | None) -> list[str]:
    """Normalise defences from a comma-separated string or list, and check they are valid.

    "none" (or an empty value) means no defences. "trust" builds on the hardened prompt,
    so it requires "prompt".
    """
    if value is None:
        return []
    items = value.split(",") if isinstance(value, str) else value
    defences = []
    for item in items:
        name = item.strip().lower()
        if name and name != "none" and name not in defences:
            defences.append(name)

    unknown = [d for d in defences if d not in VALID_DEFENCES]
    if unknown:
        raise ValueError(f"Unknown defence(s) {unknown}; valid values are {list(VALID_DEFENCES)} or 'none'")
    if "trust" in defences and "prompt" not in defences:
        raise ValueError("The 'trust' defence requires the 'prompt' defence")
    return defences


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

    # Prompt-injection defences, e.g. DEFENCES=prompt (comma-separated; empty means none)
    defences: Annotated[list[str], NoDecode] = []

    @field_validator("defences", mode="before")
    @classmethod
    def _parse_defences(cls, value):
        return parse_defences(value)


settings = Settings()