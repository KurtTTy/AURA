from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Sequence

from .base import CompletionResult, LLMProvider, Message, ProviderError

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI (or any Chat-Completions-compatible endpoint)."""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5",
        base_url: str | None = None,
        timeout: float = 120.0,
        max_tokens_param: str = "max_completion_tokens",
    ) -> None:
        if not api_key:
            raise ValueError("OpenAIProvider requires an API key")

        try:
            import openai
        except ImportError as exc:
            raise ProviderError(
                "The 'openai' package is not installed. Run: pip install openai"
            ) from exc

        self._openai = openai
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
        )
        self.model = model
        self.base_url = base_url

        # OpenAI renamed this parameter: older models take `max_tokens`,
        # newer ones require `max_completion_tokens` and reject the old
        # name. Which one applies depends on the model, so it's a setting
        # rather than a guess baked into code. Flip it in .env if you get
        # a 400 complaining about an unsupported parameter.
        self._max_tokens_param = max_tokens_param

    # ── internals ────────────────────────────────────────────────

    def _build_kwargs(
        self,
        messages: Sequence[Message],
        model: str,
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Almost a straight pass-through - see the module docstring."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs[self._max_tokens_param] = max_tokens
        return kwargs

    def _wrap_error(self, exc: Exception) -> ProviderError:
        o = self._openai
        if isinstance(exc, o.AuthenticationError):
            return ProviderError("OpenAI rejected the API key (401). Check OPENAI_API_KEY.")
        if isinstance(exc, o.NotFoundError):
            return ProviderError(
                f"Unknown model. Check OPENAI_MODEL against the /api/models list. ({exc})"
            )
        if isinstance(exc, o.RateLimitError):
            return ProviderError("OpenAI rate limit hit (429). Back off and retry.")
        if isinstance(exc, o.APIConnectionError):
            return ProviderError("Cannot reach the OpenAI API. Check your connection.")
        if isinstance(exc, o.BadRequestError):
            # By far the most likely cause here is the max_tokens rename,
            # so say so instead of echoing a bare 400.
            return ProviderError(
                f"OpenAI rejected the request: {exc}. If it mentions "
                f"'{self._max_tokens_param}', flip OPENAI_MAX_TOKENS_PARAM in .env."
            )
        if isinstance(exc, o.APIStatusError):
            return ProviderError(f"OpenAI API error {exc.status_code}: {exc}")
        return ProviderError(f"OpenAI request failed: {exc}")

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
        try:
            response = await self._client.chat.completions.create(
                **self._build_kwargs(messages, use_model, temperature, max_tokens)
            )
        except Exception as exc:
            raise self._wrap_error(exc) from exc

        choice = response.choices[0] if response.choices else None
        usage = getattr(response, "usage", None)
        return CompletionResult(
            text=(getattr(choice.message, "content", None) or "") if choice else "",
            provider=self.name,
            model=getattr(response, "model", use_model),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
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
        kwargs = self._build_kwargs(messages, use_model, temperature, max_tokens)
        kwargs["stream"] = True

        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                # `delta.content` is None on the final chunk and on
                # role-only chunks, so guard rather than yielding None.
                fragment = getattr(chunk.choices[0].delta, "content", None)
                if fragment:
                    yield fragment
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Refuses, like every cloud provider here.

        OpenAI *does* have an embeddings endpoint - this refusal is not
        about capability. Mixing embedding models inside one vector store
        silently destroys retrieval quality, so embeddings stay local and
        uniform. See cloud_provider.py for the full explanation.
        """
        raise NotImplementedError(
            "Embeddings are always local (Ollama) so the vector store stays coherent."
        )

    async def health(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        try:
            page = await self._client.models.list()
            return sorted(m.id for m in page.data)
        except Exception:
            logger.warning("Could not list OpenAI models", exc_info=True)
            return []

    async def aclose(self) -> None:
        await self._client.close()
