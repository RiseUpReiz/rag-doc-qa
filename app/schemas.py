from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask about the documents.")
    k: int | None = Field(None, ge=1, le=20, description="How many chunks to retrieve (optional).")

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "How does RAG reduce hallucination?"}]
        }
    }

class Source(BaseModel):
    source: str
    page: int | None = None

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]

