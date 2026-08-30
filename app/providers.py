from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings

def get_embeddings():
    """Get embeddings based on the configured provider."""
    if settings.embedding_provider == "google":
        if not settings.google_api_key:
            raise ValueError("Google API key is not set in the environment. Add to your .env file")
        return GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.google_api_key,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")