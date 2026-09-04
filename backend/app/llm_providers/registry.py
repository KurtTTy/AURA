from __future__ import annotations

import logging

from app.config import Settings
from .anthropic_provider import AnthropicProvider
from .base import LLMProvider, ProviderError, UnknownProviderError
from .cloud_provider import GeminiProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Holds the live provider instances for the application's lifetime."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._providers: dict[str, LLMProvider] = {}

        # Ollama is always registered. It's the default, and the only
        # provider allowed to produce embeddings.
        self._providers["ollama"] = OllamaProvider(
            base_url=settings.ollama_base_url,
            chat_model=settings.ollama_chat_model,
            embed_model=settings.ollama_embed_model,
            timeout=settings.ollama_timeout,
        )

        # ── optional cloud providers ─────────────────────────────
        # Each is wrapped: a missing SDK raises ProviderError from the
        # constructor, and that must not take down the whole app. Log it
        # and carry on with the providers that did load.
        if settings.gemini_api_key:
            self._register(
                "gemini",
                lambda: GeminiProvider(
                    api_key=settings.gemini_api_key or "",
                    model=settings.gemini_model,
                    timeout=settings.cloud_timeout,
                    enable_search=settings.gemini_enable_search,
                ),
            )

        if settings.anthropic_api_key:
            self._register(
                "anthropic",
                lambda: AnthropicProvider(
                    api_key=settings.anthropic_api_key or "",
                    model=settings.anthropic_model,
                    timeout=settings.cloud_timeout,
                    enable_fallbacks=settings.anthropic_enable_fallbacks,
                ),
            )

        if settings.openai_api_key:
            self._register(
                "openai",
                lambda: OpenAIProvider(
                    api_key=settings.openai_api_key or "",
                    model=settings.openai_model,
                    base_url=settings.openai_base_url,
                    timeout=settings.cloud_timeout,
                    max_tokens_param=settings.openai_max_tokens_param,
                ),
            )

        cloud = [name for name in self._providers if name != "ollama"]
        if cloud:
            logger.info("Cloud providers enabled: %s", ", ".join(sorted(cloud)))
        else:
            logger.info("No cloud API keys set - running fully local")

    def _register(self, name: str, build) -> None:
        """Construct a provider, tolerating a missing optional SDK."""
        try:
            self._providers[name] = build()
        except (ProviderError, ValueError) as exc:
            logger.warning("Could not enable '%s': %s", name, exc)

    @property
    def available(self) -> list[str]:
        return sorted(self._providers)

    def get(self, name: str | None = None) -> LLMProvider:
        """Return a provider by name, or the configured default.

        Raises UnknownProviderError (not KeyError) for an unknown name,
        so the API layer turns it into a clean 400 rather than a 500 or
        a misleading 503.
        """
        key = (name or self._settings.default_provider).lower()
        provider = self._providers.get(key)
        if provider is None:
            raise UnknownProviderError(
                f"Unknown or unconfigured provider '{key}'. "
                f"Available: {', '.join(self.available)}. "
                "Cloud providers need their API key set in .env."
            )
        return provider

    def embedding_provider(self) -> LLMProvider:
        """The provider used for embeddings - always local Ollama.

        Hard-coded on purpose. Embeddings are the one thing that must
        never vary by request, because the vector store's contents are
        only comparable to query vectors from the same model. Every
        cloud provider here refuses to embed for the same reason.
        """
        return self._providers["ollama"]

    async def health(self) -> dict[str, bool]:
        """Health of every registered provider, for GET /health."""
        return {name: await p.health() for name, p in self._providers.items()}

    async def list_models(self) -> dict[str, list[str]]:
        """Live model ids per provider, for GET /api/models."""
        return {name: await p.list_models() for name, p in self._providers.items()}

    def default_model_for(self, name: str) -> str | None:
        """The configured default model for a provider, if it has one."""
        return {
            "ollama": self._settings.ollama_chat_model,
            "gemini": self._settings.gemini_model,
            "anthropic": self._settings.anthropic_model,
            "openai": self._settings.openai_model,
        }.get(name)

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()
