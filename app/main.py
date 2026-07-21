from fastapi import FastAPI

from app.config import settings

app = FastAPI(title=settings.app_name)


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "app": settings.app_name}