from langchain_core.prompts import ChatPromptTemplate

from app.config import parse_defences, parse_domains, settings
from app.link_guard import guard
from app.prompts import build_hardened_messages
from app.providers import get_llm
from app.vectorstore import load_index

IMPLEMENTED_DEFENCES = ("prompt", "trust", "links")

PROMPT = ChatPromptTemplate.from_messages(
    [   
        (
            "system",
            "You are a helpful assistant that answers questions using only the "
            "provided context. If the answer is not in the context, say you don't "
            "know based on the available documents. Do not use outside knowledge."
        ),
        ("human", "{context}\n\nQuestion: {question}"),
    ]
)

def format_context(docs) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

def extract_text(response) -> str:
    """Pull plain text out of a model response, whether it's a string or content parts."""
    content = response.content
    if isinstance(content, str):
        return content
    parts = [part.get("text", "") for part in content if isinstance(part, dict)]
    return "".join(parts).strip()

def check_defences(defences: list[str]) -> list[str]:
    """Validate a defences list and fail on any that aren't implemented yet."""
    defences = parse_defences(defences)
    pending = [d for d in defences if d not in IMPLEMENTED_DEFENCES]
    if pending:
        raise NotImplementedError(f"Defence(s) not implemented yet: {pending}")
    return defences

def answer_question(
    question: str,
    k: int | None = None,
    store=None,
    defences: list[str] | None = None,
    allowed_domains: list[str] | None = None,
) -> dict:
    """Retrieve relevant chunks, ground an answer in them, and return sources.

    defences defaults to settings.defences. With "prompt" on, the hardened prompt is used;
    otherwise the original prompt is used unchanged. With "trust" on too, each chunk's trust
    level (set at ingestion from the manifest) is shown to the model. With "links" on, links
    and emails outside allowed_domains (default settings.allowed_link_domains) are removed
    from the answer and listed in links_removed; with it off, links_removed is None.
    """
    defences = check_defences(settings.defences if defences is None else defences)
    if store is None:
        store = load_index()
    docs = store.similarity_search(question, k=k or settings.retriever_k)

    if "prompt" in defences:
        prompt = build_hardened_messages(docs, question, trust="trust" in defences)
    else:
        prompt = PROMPT.invoke({"context": format_context(docs), "question": question})
    response = get_llm().invoke(prompt)

    sources = []
    for doc in docs:
        source = {"source": doc.metadata.get("source")}
        if "page" in doc.metadata:
            source["page"] = doc.metadata["page"]
        if source not in sources:
            sources.append(source)

    answer = extract_text(response)
    links_removed = None
    if "links" in defences:
        allowed = settings.allowed_link_domains if allowed_domains is None else parse_domains(allowed_domains)
        source_names = [s["source"] for s in sources if s["source"]]
        answer, links_removed = guard(answer, allowed, source_names)

    return {"answer": answer, "sources": sources, "links_removed": links_removed}

if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"
    result = answer_question(question)
    print(f"\nQuestion: {question}\n")
    print(f"Answer: {result['answer']}\n")
    print("Sources:")
    for source in result["sources"]:
        print("-", source)