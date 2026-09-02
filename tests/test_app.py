from langchain_core.documents import Document
from fastapi.testclient import TestClient

from app.ingest import chunk_documents
from app.rag import extract_text
from app.main import app

client = TestClient(app)

def test_split_documents_preserves_metadata_and_splits():
    doc = Document(page_content="Leave policy details. " * 200,
                   metadata={"source": "policy.pdf", "page":1})
    chunks = chunk_documents([doc])
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "policy.pdf" for c in chunks)

def test_extract_text_handles_plain_string():
    class R:
        content = "hello"
    assert extract_text(R()) == "hello"

def test_extract_text_handles_content_parts():
    class R:
        content = [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]
    assert extract_text(R()) == "part one part two"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_ask_returns_answers_and_sources(monkeypatch):
    def fake_answer(question, k=None):
        return {"answer": "A grounded answer.", "sources": [{"source": "doc.pdf", "page": 2}]}
    monkeypatch.setattr("app.main.answer_question", fake_answer)
    response = client.post("/ask", json={"question": "What is in the doc?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "A grounded answer."
    assert body["sources"][0] == {"source": "doc.pdf", "page": 2}

def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422