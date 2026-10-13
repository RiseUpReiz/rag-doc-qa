from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from app.config import settings
from app.rag import answer_question
from app.schemas import AskRequest, AskResponse
from app.trust import load_manifest
from app.vectorstore import build_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app.

    With the "trust" defence on, a missing or invalid manifest stops startup instead of
    failing on the first question.
    """
    if "trust" in settings.defences:
        load_manifest(Path(settings.trusted_sources_path))
    if not Path(settings.persist_dir).exists():
        build_index()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

@app.get("/", include_in_schema=False)
def root():
    """Send visitors to the interactive API docs."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "app": settings.app_name}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question using the ingested documents, with sources."""
    try:
        result = answer_question(request.question, k=request.k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(**result)