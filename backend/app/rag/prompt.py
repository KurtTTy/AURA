from __future__ import annotations

from app.llm_providers import Message
from app.rag.retrieve import RetrievedChunk

#: Retrieval found nothing. A constant, so the wording is consistent
NO_CONTEXT_ANSWER = (
    "I don't have any documents indexed that relate to that question. "
    "Try ingesting relevant documents first."
)

SYSTEM_PROMPT = (
    "You are a document-answering assistant. Answer only using the "
    "information in the CONTEXT section provided by the user. "
    "Do not use outside knowledge. "
    "If the context does not contain the answer, reply exactly: "
    "The provided documents do not cover this. "
    "Cite the source for every claim using its label, like [1] or [2]. "
    "Keep answers concise."
)

def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks into one labelled string, most relevant first.

    Labels let the model cite "[1]"; the rule between chunks stops it
    treating them as one document.
    """
    if not chunks:
        return ""

    blocks = []
    for position, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{position}] source: {chunk.source} (chunk {chunk.chunk_index})\n"
            f"{chunk.text}"
        )

    return "\n\n---\n\n".join(blocks)


def build_rag_messages(question: str, chunks: list[RetrievedChunk], history: list[Message] | None = None,) -> list[Message]:
    """Build [system] + [history] + [user turn with context].

    Context goes in the USER message: the system prompt is stable
    instructions, the context changes per question.
    """
    context = format_context(chunks)
    user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    messages = [Message(role="system", content=SYSTEM_PROMPT)]

    if history:
        messages += history[-6:]
    messages.append(Message(role="user", content=user_content))

    return messages

