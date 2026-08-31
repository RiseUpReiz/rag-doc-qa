from fastapi import FastAPI, HTTPException

from app.config import settings
from app.rag import answer_question
from app.schemas import AskRequest, AskResponse

app = FastAPI(title=settings.app_name)


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