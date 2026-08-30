from pathlib import Path

from langchain_chroma import Chroma

from app.config import settings
from app.ingest import ingest
from app.providers import get_embeddings

def build_index() -> Chroma:
    """Embed all document chunks and persist them to the Chroma vector store."""
    chunks = ingest()
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=settings.persist_dir,
        collection_name=settings.collection_name
    )

def load_index() -> Chroma:
    """Open the existing persisteed Chroma vector store."""
    return Chroma(
        embedding_function=get_embeddings(),
        persist_directory=settings.persist_dir,
        collection_name=settings.collection_name
    )

if __name__ == "__main__":
    import sys

    if Path(settings.persist_dir).exists():
        print(f"Loading existing index from...")
        store = load_index()
    else:
        print(f"Building index (embedding all chunks - this calls the Gemini API)")
        store = build_index()

    question = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"
    print(f"\nQuery: {question}\n")
    for doc in store.similarity_search(question, k=3):
        print("-" * 60)
        print(doc.metadata)
        print(doc.page_content[:200].strip(), "...")
