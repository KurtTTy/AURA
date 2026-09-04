from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.deps import get_registry
from app.llm_providers import Message, ProviderError, ProviderRegistry, UnknownProviderError
from app.models import Mode, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Answer directly. "
    "If you are unsure about a fact, say so rather than guessing."
)


def build_chat_messages(request: QueryRequest) -> list[Message]:
    """Assemble system prompt + history + the current question.

    Note the type conversion: request.history holds `ChatMessage`
    (Pydantic, validated at the HTTP boundary) while providers expect
    `Message` (a lightweight internal dataclass). Same shape, different
    purpose. You'll do this same conversion in answer_question().
    """
    messages = [Message(role="system", content=DEFAULT_SYSTEM_PROMPT)]
    messages.extend(Message(role=m.role, content=m.content) for m in request.history)
    messages.append(Message(role="user", content=request.question))
    return messages


@router.post("", response_model=QueryResponse)
async def chat(
    request: QueryRequest,
    registry: ProviderRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    """Single-turn conversation with optional history."""
    try:
        provider = registry.get(request.provider)
        result = await provider.chat(build_chat_messages(request), model=request.model)
    except UnknownProviderError as exc:
        # A bad provider name is the caller's mistake, not an outage -
        # retrying changes nothing, so 503 would be misleading advice.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return QueryResponse(
        answer=result.text,
        mode=Mode.CHAT,
        provider=result.provider,
        model=result.model,
        sources=[],
    )


@router.post("/stream")
async def chat_stream(
    request: QueryRequest,
    registry: ProviderRegistry = Depends(get_registry),
) -> StreamingResponse:
    """Stream a reply token-by-token as plain text.

    This is what makes a client feel responsive: text
    appears as it's generated instead of after a 20-second pause.

    Streaming has one awkward property worth knowing now: HTTP status
    and headers are sent before generation starts, so an error occurring
    mid-stream cannot become a 503. All we can do is emit an error
    marker into the body - which is why the generator below catches
    ProviderError and yields text rather than raising.
    """

    # Resolve the provider BEFORE the response starts. Anything raised
    # once streaming is underway can no longer become an HTTP status -
    # the 200 and its headers are already on the wire. A bad provider
    # name is knowable up front, so check it here and return a real 400.
    try:
        provider = registry.get(request.provider)
    except UnknownProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def generate():
        try:
            async for fragment in provider.chat_stream(
                build_chat_messages(request), model=request.model
            ):
                yield fragment
        except ProviderError as exc:
            # Mid-stream failures genuinely cannot be a status code, so
            # the only option left is a marker in the body.
            logger.error("Stream failed: %s", exc)
            yield f"\n\n[error] {exc}"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
