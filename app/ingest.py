from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.trust import trust_level

DATA_DIR = Path("data")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_pdf(path: Path) -> list[Document]:
    """Load a PDF into one Document per page, tagged with source and page number."""
    reader = PdfReader(path)
    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": path.name, "page": page_number},
        )
        for page_number, page in enumerate(reader.pages, start=1)
    ]


def load_text(path: Path) -> list[Document]:
    """Load a plain text file into a single Document tagged with its source."""
    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": path.name})]


def document_paths(source_dir: Path) -> list[Path]:
    """The PDF and text files in source_dir that ingestion will load, in a stable order."""
    return [
        path for path in sorted(source_dir.iterdir())
        if path.is_file() and path.suffix.lower() in (".pdf", ".txt", ".md")
    ]


def load_documents(source_dir: Path, manifest: dict[str, str] | None = None) -> list[Document]:
    """Load every PDF and text file in source_dir into a list of Documents.

    With a trust manifest, each Document also gets metadata["trust"] ("official" or
    "unverified"), which its chunks inherit when split.
    """
    documents: list[Document] = []
    for path in document_paths(source_dir):
        loaded = load_pdf(path) if path.suffix.lower() == ".pdf" else load_text(path)
        if manifest is not None:
            level = trust_level(path, manifest)
            for doc in loaded:
                doc.metadata["trust"] = level
        documents.extend(loaded)
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split Documents into overlapping chunks sized for embedding and retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def ingest(source_dir: Path = DATA_DIR, manifest: dict[str, str] | None = None) -> list[Document]:
    """Load every document in source_dir and split it into embeddable chunks."""
    return chunk_documents(load_documents(source_dir, manifest))


if __name__ == "__main__":
    documents = load_documents(DATA_DIR)
    chunks = chunk_documents(documents)

    print(f"Loaded {len(documents)} document(s) from {DATA_DIR}/")
    print(f"Split into {len(chunks)} chunk(s)")
    print()
    print("First chunk:")
    print(f"  metadata: {chunks[0].metadata}")
    print(f"  content:  {chunks[0].page_content[:200]!r}")
