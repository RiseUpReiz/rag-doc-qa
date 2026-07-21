from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "RAG Document Q&A"
    # Provider keys — unused in Stage 1, wired up from Stage 2 onward.
    google_api_key: str | None = None


settings = Settings()