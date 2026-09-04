from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Sequence

Role = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class Message:
    """One turn in a conversation. Internal transport type; the API
    boundary uses Pydantic instead (app/models/schemas.py)."""

    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class CompletionResult:
    """A finished generation. `provider`/`model` are what ACTUALLY ran,
    which can differ from what was requested."""

    text: str
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    #: Web pages the provider searched server-side. NOT the same as a RAG
    #: answer's sources, which are chunks from your own documents.
    sources: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """A provider failed. Server-side condition; the API maps it to 503."""


class UnknownProviderError(ProviderError):
    """Unregistered provider name. A CLIENT error - retrying changes
    nothing - so the API maps it to 400, not 503. Subclasses
    ProviderError, so catch this one first."""


class LLMProvider(ABC):
    """Abstract base every provider implements.

    Cloud providers raise NotImplementedError from embed() on purpose -
    see cloud_provider.py.
    """

    #: Short identifier, e.g. "ollama". Reported in CompletionResult.
    name: str

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Generate a complete response.

        `messages` is oldest-first; a system prompt is just the first
        message with role="system". Default temperature is 0.2 - grounded
        answers want faithfulness, not creativity.

        Raises:
            ProviderError: provider unreachable or erroring.
        """

    @abstractmethod
    def chat_stream(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield the response in fragments.

        Concatenating every fragment must equal what chat() returns.
        Declared as a normal method, not `async def` - implementations
        are async generators, so calling it runs nothing until iterated.
        """

    @abstractmethod
    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Turn texts into vectors, one per input, same order, same width.

        Batch: 200 chunks in one call beats 200 calls.

        Raises:
            NotImplementedError: provider has no compatible embedder.
            ProviderError: transport or API failure.
        """

    @abstractmethod
    async def health(self) -> bool:
        """Is this provider usable right now?

        Must not raise - catch everything and return False. A health check
        that throws cannot be used to decide on fallback.
        """

    async def list_models(self) -> list[str]:
        """Model ids this provider can serve, queried live.

        Must not raise; [] means "I don't know". Hard-coded catalogues go
        stale within months.
        """
        return []

    async def aclose(self) -> None:
        """Release connection pools. Called from the lifespan shutdown."""
        return None
