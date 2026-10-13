import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from app import ingest, main, rag, vectorstore
from app.trust import ManifestError, hash_file
from app.vectorstore import EmptyIndexError


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "policy.md").write_text("Staff get 22 days of annual leave.\n", encoding="utf-8")
    (data / "notice.md").write_text("Unapproved notice.\n", encoding="utf-8")
    monkeypatch.setattr(vectorstore, "ingest", lambda manifest=None: ingest.ingest(data, manifest))
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: None)
    stored = []
    monkeypatch.setattr(vectorstore.Chroma, "from_documents", lambda documents, **kwargs: stored.extend(documents))
    return data, stored


def test_build_index_tags_chunks_with_trust_when_trust_is_on(corpus, tmp_path, monkeypatch):
    data, stored = corpus
    manifest = tmp_path / "trusted_sources.json"
    manifest.write_text(json.dumps({"official": {"policy.md": hash_file(data / "policy.md")}}), encoding="utf-8")
    monkeypatch.setattr(vectorstore.settings, "defences", ["prompt", "trust", "links"])
    monkeypatch.setattr(vectorstore.settings, "trusted_sources_path", str(manifest))

    vectorstore.build_index()

    assert {c.metadata["source"]: c.metadata["trust"] for c in stored} == {
        "policy.md": "official", "notice.md": "unverified",
    }


def test_build_index_fails_closed_without_manifest_when_trust_is_on(corpus, tmp_path, monkeypatch):
    _, stored = corpus
    monkeypatch.setattr(vectorstore.settings, "defences", ["prompt", "trust", "links"])
    monkeypatch.setattr(vectorstore.settings, "trusted_sources_path", str(tmp_path / "missing.json"))

    with pytest.raises(ManifestError, match="make-manifest"):
        vectorstore.build_index()
    assert stored == []


def test_build_index_without_trust_and_without_manifest_leaves_chunks_untagged(corpus, tmp_path, monkeypatch):
    _, stored = corpus
    monkeypatch.setattr(vectorstore.settings, "defences", ["prompt"])
    monkeypatch.setattr(vectorstore.settings, "trusted_sources_path", str(tmp_path / "missing.json"))

    vectorstore.build_index()

    assert stored and all("trust" not in c.metadata for c in stored)


# --- empty index ---

class CountingEmbeddings(DeterministicFakeEmbedding):
    calls: int = 0

    def embed_query(self, text):
        self.calls += 1
        return super().embed_query(text)


@pytest.fixture
def empty_storage(tmp_path, monkeypatch):
    """A persist_dir that exists but holds an empty collection, as opening an unbuilt index leaves it."""
    embeddings = CountingEmbeddings(size=8)
    monkeypatch.setattr(vectorstore.settings, "persist_dir", str(tmp_path / "storage"))
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: embeddings)
    vectorstore.load_index()
    assert (tmp_path / "storage").exists()
    return embeddings


def test_index_needs_build_when_storage_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(vectorstore.settings, "persist_dir", str(tmp_path / "storage"))
    assert vectorstore.index_needs_build()


def test_existing_but_empty_collection_needs_build(empty_storage):
    assert vectorstore.index_needs_build()
    vectorstore.load_index().add_documents([Document(page_content="22 days.", metadata={"source": "a.md"})])
    assert not vectorstore.index_needs_build()


def test_app_startup_builds_an_existing_but_empty_index(empty_storage, monkeypatch):
    built = []
    monkeypatch.setattr(main.settings, "defences", [])
    monkeypatch.setattr(main, "build_index", lambda: built.append(True))
    with TestClient(main.app):
        pass
    assert built == [True]


def test_answer_question_raises_on_empty_index_before_any_model_call(empty_storage, monkeypatch):
    monkeypatch.setattr(rag, "load_index", vectorstore.load_index)
    monkeypatch.setattr(rag, "get_llm", lambda: pytest.fail("LLM called with an empty index"))
    with pytest.raises(EmptyIndexError, match=r"python -m app\.vectorstore"):
        rag.answer_question("How many days?", defences=[])
    assert empty_storage.calls == 0


def test_answer_question_raises_when_a_given_store_returns_nothing(monkeypatch):
    class EmptyStore:
        def similarity_search(self, question, k):
            return []

    monkeypatch.setattr(rag, "get_llm", lambda: pytest.fail("LLM called with no documents"))
    with pytest.raises(EmptyIndexError, match=r"python -m app\.vectorstore"):
        rag.answer_question("How many days?", store=EmptyStore(), defences=[])
