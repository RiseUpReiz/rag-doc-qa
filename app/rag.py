from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.providers import get_llm
from app.vectorstore import load_index

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

def answer_question(question: str, k: int | None = None, store=None) -> dict:
    """Retrieve relevant chunks, ground an answer in them, and return sources."""
    if store is None:
        store = load_index()
    docs = store.similarity_search(question, k=k or settings.retriever_k)

    prompt = PROMPT.invoke({"context": format_context(docs), "question": question})
    response = get_llm().invoke(prompt)

    sources = []
    for doc in docs:
        source = {"source": doc.metadata.get("source")}
        if "page" in doc.metadata:
            source["page"] = doc.metadata["page"]
        if source not in sources:
            sources.append(source)

    return {"answer": extract_text(response), "sources": sources}

if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"
    result = answer_question(question)
    print(f"\nQuestion: {question}\n")
    print(f"Answer: {result['answer']}\n")
    print("Sources:")
    for source in result["sources"]:
        print("-", source)