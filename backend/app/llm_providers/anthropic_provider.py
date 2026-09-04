from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Sequence

from .base import CompletionResult, LLMProvider, Message, ProviderError

logger = logging.getLogger(__name__)

#: Anthropic requires max_tokens on every request. These are the sizes
#: the Anthropic docs recommend as defaults: keep non-streaming under the
#: SDK's HTTP timeout, and give streaming room to breathe.
DEFAULT_MAX_TOKENS = 16_000
DEFAULT_STREAM_MAX_TOKENS = 64_000

#: Model families that REMOVED the sampling parameters. Sending
#: temperature/top_p/top_k to any of these returns HTTP 400 - it is not
#: ignored. Our LLMProvider interface takes a `temperature` argument for
#: every provider, so we filter it out here rather than making callers
#: care which backend they're talking to.
_NO_SAMPLING_PARAMS = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)


def _accepts_sampling(model: str) -> bool:
    """Whether `model` still accepts temperature/top_p/top_k."""
    return not model.startswith(_NO_SAMPLING_PARAMS)


class AnthropicProvider(LLMProvider):
    """Claude models via the official `anthropic` async SDK."""

    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-5",
        timeout: float = 120.0,
        enable_fallbacks: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("AnthropicProvider requires an API key")

        # Imported lazily so the package is only needed when this
        # provider is actually configured. Someone running fully local
        # should never have to install it.
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError(
                "The 'anthropic' package is not installed. Run: pip install anthropic"
            ) from exc

        self._anthropic = anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)
        self.model = model

        # Claude's safety classifiers can decline a request and return a
        # normal 200 with stop_reason="refusal" and no content. Enabling
        # server-side fallbacks makes the API retry on another model
        # instead, so a declined request still produces an answer.
        # Toggleable because it rides a beta header that not every API
        # key has access to - set ANTHROPIC_ENABLE_FALLBACKS=false if
        # you get a 400 mentioning the beta.
        self._enable_fallbacks = enable_fallbacks

    # ── internals ────────────────────────────────────────────────

    def _messages_api(self):
        """The beta namespace when fallbacks are on, the stable one otherwise."""
        return self._client.beta.messages if self._enable_fallbacks else self._client.messages

    def _build_kwargs(
        self,
        messages: Sequence[Message],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Translate our Message list into Anthropic's request shape.

        Two differences from the OpenAI/Ollama convention, both absorbed
        here so nothing upstream has to know:

          1. The system prompt is a top-level `system` string, NOT a
             message with role="system". (Same as Gemini - see
             cloud_provider.py - though Gemini spells it differently
             again, which is exactly why this layer exists.)
          2. Only "user" and "assistant" roles are valid in `messages`.
        """
        system_parts: list[str] = []
        turns: list[dict[str, str]] = []

        for message in messages:
            if message.role == "system":
                system_parts.append(message.content)
            else:
                turns.append({"role": message.role, "content": message.content})

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": turns,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        # Only send temperature where the model still accepts it.
        if _accepts_sampling(model):
            kwargs["temperature"] = temperature

        if self._enable_fallbacks:
            # "default" lets Anthropic pick the fallback model by refusal
            # category, so there's no model list here to go stale.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"

        return kwargs

    def _wrap_error(self, exc: Exception) -> ProviderError:
        """Map the SDK's typed exceptions onto our single error type."""
        a = self._anthropic
        if isinstance(exc, a.AuthenticationError):
            return ProviderError("Anthropic rejected the API key (401). Check ANTHROPIC_API_KEY.")
        if isinstance(exc, a.NotFoundError):
            return ProviderError(
                f"Unknown Anthropic model. Check ANTHROPIC_MODEL against the /api/models list. ({exc})"
            )
        if isinstance(exc, a.RateLimitError):
            return ProviderError("Anthropic rate limit hit (429). Back off and retry.")
        if isinstance(exc, a.APIConnectionError):
            return ProviderError("Cannot reach the Anthropic API. Check your connection.")
        if isinstance(exc, a.APIStatusError):
            return ProviderError(f"Anthropic API error {exc.status_code}: {exc.message}")
        return ProviderError(f"Anthropic request failed: {exc}")

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull text out of the content-block list.

        `content` is a list of blocks (text, thinking, tool_use, ...), so
        we filter for text rather than assuming content[0].text - which
        breaks outright on a refusal, where content is empty.
        """
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )

    # ── LLMProvider implementation ───────────────────────────────

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        use_model = model or self.model
        kwargs = self._build_kwargs(
            messages, use_model, temperature, max_tokens or DEFAULT_MAX_TOKENS
        )

        try:
            response = await self._messages_api().create(**kwargs)
        except Exception as exc:
            raise self._wrap_error(exc) from exc

        # Check stop_reason BEFORE reading content. A safety refusal is a
        # successful HTTP 200 with an empty content list, so code that
        # goes straight to the text gets "" and no explanation.
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise ProviderError(
                f"Claude declined this request (safety category: {category}). "
                "Rephrase, or use the local Ollama provider."
            )

        usage = getattr(response, "usage", None)
        return CompletionResult(
            text=self._extract_text(response),
            provider=self.name,
            # response.model is what ACTUALLY served it, which may differ
            # from what we asked for when a fallback kicked in.
            model=getattr(response, "model", use_model),
            prompt_tokens=getattr(usage, "input_tokens", None),
            completion_tokens=getattr(usage, "output_tokens", None),
        )

    async def chat_stream(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        use_model = model or self.model
        kwargs = self._build_kwargs(
            messages, use_model, temperature, max_tokens or DEFAULT_STREAM_MAX_TOKENS
        )

        try:
            async with self._messages_api().stream(**kwargs) as stream:
                # text_stream yields only the text deltas, skipping the
                # thinking/tool blocks we don't surface here.
                async for fragment in stream.text_stream:
                    yield fragment
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Anthropic has no embeddings endpoint - and we wouldn't use one
        anyway. Embeddings stay local so the vector store keeps a single
        coordinate space; see cloud_provider.py for the full reasoning."""
        raise NotImplementedError(
            "Anthropic does not provide embeddings. Embeddings are always local (Ollama)."
        )

    async def health(self) -> bool:
        try:
            await self._client.models.list(limit=1)
            return True
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        try:
            page = await self._client.models.list(limit=100)
            return [m.id for m in page.data]
        except Exception:
            logger.warning("Could not list Anthropic models", exc_info=True)
            return []

    async def aclose(self) -> None:
        await self._client.close()
