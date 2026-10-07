from pathlib import Path

from langchain_chroma import Chroma

from app.config import settings
from app.ingest import ingest
from app.providers import get_embeddings
from app.trust import load_manifest_if_needed

def build_index() -> Chroma:
    """Embed all document chunks and persist them to the Chroma vector store.

    If a trust manifest exists, chunks are tagged with their trust level. With the "trust"
    defence on, the manifest is required: a missing or invalid one raises instead of
    leaving documents untagged.
    """
    manifest = load_manifest_if_needed(
        Path(settings.trusted_sources_path), required="trust" in settings.defences
    )
    chunks = ingest(manifest=manifest)
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
