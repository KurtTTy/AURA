from __future__ import annotations

import pytest

from app.config import get_settings
from app.llm_providers import (
    GeminiProvider,
    Message,
    OllamaProvider,
    ProviderRegistry,
)


class TestMessageTranslation:
    """Provider-shape translation, verified without any network call."""

    def test_message_to_dict(self):
        assert Message(role="user", content="hi").to_dict() == {
            "role": "user",
            "content": "hi",
        }

    def test_gemini_renames_assistant_to_model(self):
        """Gemini calls the assistant role "model". The provider must
        translate so callers never have to know."""
        payload = GeminiProvider._to_gemini_payload(
            [Message(role="user", content="q"), Message(role="assistant", content="a")],
            temperature=0.2,
            max_tokens=None,
        )
        assert [c["role"] for c in payload["contents"]] == ["user", "model"]

    def test_gemini_lifts_system_prompt_out_of_contents(self):
        """Gemini takes system prompts in a separate top-level field,
        not as a message in the list."""
        payload = GeminiProvider._to_gemini_payload(
            [Message(role="system", content="be terse"), Message(role="user", content="q")],
            temperature=0.2,
            max_tokens=None,
        )
        assert payload["systemInstruction"]["parts"][0]["text"] == "be terse"
        assert len(payload["contents"]) == 1


class TestAnthropicRules:
    """Claude-specific request rules, verified without the SDK installed.

    `_accepts_sampling` is a pure function, so these run offline and
    regardless of whether `anthropic` is installed.
    """

    @pytest.mark.parametrize(
        "model",
        ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-fable-5"],
    )
    def test_current_models_reject_temperature(self, model):
        """Sending temperature to these returns a 400 - it is not ignored.

        Our LLMProvider interface takes `temperature` for every provider,
        so the Anthropic provider has to strip it rather than making
        callers know which backend they're on.
        """
        from app.llm_providers.anthropic_provider import _accepts_sampling

        assert not _accepts_sampling(model)

    @pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-sonnet-4-5"])
    def test_older_models_still_accept_temperature(self, model):
        from app.llm_providers.anthropic_provider import _accepts_sampling

        assert _accepts_sampling(model)


class TestCatalog:
    def test_every_curated_model_has_guidance(self):
        from app.llm_providers import all_recommended

        for info in all_recommended():
            assert info.notes and info.context and info.cost, info.id

    def test_describe_ignores_ollama_latest_tag(self):
        """Ollama reports an untagged model as 'name:latest'."""
        from app.llm_providers import describe

        assert describe("ollama", "nomic-embed-text:latest") is not None
        assert describe("ollama", "nomic-embed-text") is not None

    def test_describe_returns_none_for_unknown(self):
        from app.llm_providers import describe

        assert describe("ollama", "not-a-real-model") is None

    def test_openai_and_gemini_are_live_only(self):
        """Their ids change too often to hard-code - fetched at runtime."""
        from app.llm_providers.catalog import LIVE_ONLY_PROVIDERS, RECOMMENDED

        for name in LIVE_ONLY_PROVIDERS:
            assert name not in RECOMMENDED


class TestRegistry:
    def test_ollama_always_registered(self):
        registry = ProviderRegistry(get_settings())
        assert "ollama" in registry.available

    def test_cloud_providers_absent_without_keys(self):
        """The default posture is fully local: no key, no provider, and
        no optional SDK required."""
        settings = get_settings()
        registry = ProviderRegistry(settings)

        for name, key in (
            ("anthropic", settings.anthropic_api_key),
            ("openai", settings.openai_api_key),
            ("gemini", settings.gemini_api_key),
        ):
            if not key:
                assert name not in registry.available

    def test_unknown_provider_raises_provider_error(self):
        from app.llm_providers import ProviderError

        registry = ProviderRegistry(get_settings())
        # The message also covers the "configured but no API key" case,
        # which is the more common mistake once cloud providers exist.
        with pytest.raises(ProviderError, match="Unknown or unconfigured provider"):
            registry.get("does-not-exist")

    def test_embedding_provider_is_always_ollama(self):
        """Embeddings must never vary by request - see cloud_provider.py."""
        registry = ProviderRegistry(get_settings())
        assert registry.embedding_provider().name == "ollama"


class TestErrorSemantics:
    """An unknown provider is a CLIENT error, not an outage.

    503 means "try again later", which is misleading advice when the
    request will fail identically forever. These run offline - the
    registry rejects the name before any network call happens.
    """

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from app.main import app

        with TestClient(app) as c:
            yield c

    @pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
    def test_unknown_provider_is_400(self, client, path):
        """Chat paths validate the provider before touching anything."""
        response = client.post(path, json={"question": "x", "provider": "nope"})
        assert response.status_code == 400, (
            f"{path} returned {response.status_code} for an unknown provider; "
            "expected 400 (bad request), not 503 (service unavailable)"
        )

    @pytest.mark.integration
    @pytest.mark.parametrize("path", ["/api/rag/query", "/api/query"])
    def test_unknown_provider_is_400_on_rag_paths(self, client, path):
        """Same rule, but these genuinely need Ollama running.

        `answer_question()` retrieves BEFORE it resolves the provider, and
        retrieval embeds the question - which is a local Ollama call. With
        Ollama stopped these fail at the embedding step (503) and never
        reach the provider check, so the assertion below would be testing
        the environment rather than the routing.

        Marked integration for that reason. The two chat paths above cover
        the same behaviour offline.
        """
        response = client.post(path, json={"question": "x", "provider": "nope"})
        assert response.status_code == 400, (
            f"{path} returned {response.status_code} for an unknown provider; "
            "expected 400 (bad request), not 503 (service unavailable)"
        )

    def test_the_message_lists_what_is_available(self, client):
        detail = client.post(
            "/api/chat", json={"question": "x", "provider": "nope"}
        ).json()["detail"]
        assert "ollama" in detail, "The error should name the providers that do exist"

    def test_missing_question_is_422(self, client):
        """Schema validation, handled by FastAPI before our code runs."""
        assert client.post("/api/chat", json={}).status_code == 422


@pytest.mark.integration
class TestOllamaLive:
    """Require a running Ollama with qwen2.5:7b + nomic-embed-text."""

    @pytest.fixture
    def provider(self):
        settings = get_settings()
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            chat_model=settings.ollama_chat_model,
            embed_model=settings.ollama_embed_model,
        )

    async def test_health(self, provider):
        assert await provider.health(), (
            "Ollama unreachable or models missing. Run: ollama list"
        )

    async def test_chat_returns_text(self, provider):
        result = await provider.chat(
            [Message(role="user", content="Reply with exactly: OK")],
            max_tokens=10,
        )
        assert result.text.strip()
        assert result.provider == "ollama"
        await provider.aclose()

    async def test_embed_dimensions_are_consistent(self, provider):
        """All vectors must share one dimensionality, or the vector
        store cannot compare them."""
        vectors = await provider.embed(["first text", "second text"])
        assert len(vectors) == 2
        assert len(vectors[0]) == len(vectors[1])
        assert len(vectors[0]) > 0
        await provider.aclose()

    async def test_stream_concatenates_to_full_answer(self, provider):
        fragments = []
        async for fragment in provider.chat_stream(
            [Message(role="user", content="Count: 1 2 3")], max_tokens=20
        ):
            fragments.append(fragment)
        assert "".join(fragments).strip()
        await provider.aclose()
