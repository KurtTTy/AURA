from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

import pytest

from app.config import get_settings
from app.llm_providers import CompletionResult, LLMProvider, Message
from app.models import Mode, QueryRequest
from app.rag.prompt import NO_CONTEXT_ANSWER
from app.routers.rag import answer_question


class FakeProvider(LLMProvider):
    """Records what it was asked, returns a fixed answer."""

    name = "fakeprovider"

    def __init__(self) -> None:
        self.chat_calls = 0
        self.last_messages: list[Message] = []

    async def chat(self, messages, *, model=None, temperature=0.2, max_tokens=None):
        self.chat_calls += 1
        self.last_messages = list(messages)
        # Deliberately different from anything configured, so a response
        # echoing the *request* instead of the *result* stands out.
        return CompletionResult(
            text="Employees are entitled to 90 days of paid parental leave [1].",
            provider="fakeprovider",
            model="fake-model-v1",
        )

    async def chat_stream(self, messages, *, model=None, temperature=0.2, max_tokens=None):
        yield "unused"

    async def embed(self, texts: Sequence[str], *, model: str | None = None):
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def health(self) -> bool:
        return True


class FakeRegistry:
    """Stands in for ProviderRegistry.

    Must mirror every method answer_question() is allowed to call. If a
    correct implementation reaches for something missing here, the test
    fails with an AttributeError that looks like a bug in the code under
    test - so this fake has to keep up with the real registry's surface.
    """

    def __init__(self, provider: FakeProvider) -> None:
        self._provider = provider

    def get(self, name: str | None = None) -> FakeProvider:
        return self._provider

    def embedding_provider(self) -> FakeProvider:
        return self._provider

    def default_model_for(self, name: str | None = None) -> str:
        """The configured default model for a provider.

        The real one returns `str | None`; this always returns a string,
        so a test failure means the code didn't call it, not that it got
        an awkward value back.
        """
        return "fake-default-model"


class FakeStore:
    """Stands in for VectorStore - only the two methods retrieve() uses."""

    collection_name = "faketest"

    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self._hits = hits or []

    def count(self) -> int:
        return len(self._hits)

    def query(self, embedding, top_k: int = 5, where=None):
        self.last_top_k = top_k
        return self._hits[:top_k]


def hit(text: str, source: str, index: int, score: float) -> dict[str, Any]:
    """One row in the shape VectorStore.query() returns."""
    return {
        "id": f"{source}::{index}",
        "text": text,
        "metadata": {"source": source, "chunk_index": index},
        "distance": 1.0 - score,
        "score": score,
    }


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def registry(provider: FakeProvider) -> FakeRegistry:
    return FakeRegistry(provider)


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def full_store() -> FakeStore:
    return FakeStore([
        hit("New parents are entitled to 90 days of paid parental leave.", "handbook.md", 0, 0.746),
        hit("Hotel accommodation is reimbursed up to 75 per night.", "expenses.md", 2, 0.577),
    ])


class TestEmptyStore:
    """The path healthcheck.py can never reach.

    Its store always contains the sample handbook, so `chunks` is never
    empty there. This is the case a real user hits on day one: asking a
    question before ingesting anything.
    """

    async def test_returns_a_valid_response(self, registry, settings):
        """Must build a complete QueryResponse - not raise.

        `provider` and `model` on QueryResponse are required strings.
        Passing `request.provider` straight through fails validation when
        the caller didn't name one, which is the normal case.
        """
        response = await answer_question(
            QueryRequest(question="anything"), FakeStore([]), registry, settings
        )
        assert response.answer == NO_CONTEXT_ANSWER
        assert response.sources == []
        assert response.mode is Mode.RAG

    async def test_reports_a_provider_and_model(self, registry, settings):
        """Both are required strings - neither may be None."""
        response = await answer_question(
            QueryRequest(question="anything"), FakeStore([]), registry, settings
        )
        assert isinstance(response.provider, str) and response.provider
        assert isinstance(response.model, str) and response.model

    async def test_does_not_call_the_llm(self, provider, registry, settings):
        """The whole point of the early return: a model with no evidence
        invents an answer."""
        await answer_question(
            QueryRequest(question="anything"), FakeStore([]), registry, settings
        )
        assert provider.chat_calls == 0, "Called the LLM with no context"


class TestGroundedAnswer:
    async def test_returns_the_models_answer(self, registry, settings, full_store):
        response = await answer_question(
            QueryRequest(question="How much parental leave?"), full_store, registry, settings
        )
        assert "90 days" in response.answer
        assert response.mode is Mode.RAG

    async def test_sources_are_populated(self, registry, settings, full_store):
        """An answer without sources is indistinguishable from a guess."""
        response = await answer_question(
            QueryRequest(question="How much parental leave?"), full_store, registry, settings
        )
        assert len(response.sources) == 2
        assert {s.source for s in response.sources} == {"handbook.md", "expenses.md"}
        assert response.sources[0].chunk_index == 0

    async def test_reports_what_actually_ran(self, registry, settings, full_store):
        """Not what was requested - they can differ (cloud fallbacks)."""
        response = await answer_question(
            QueryRequest(question="q", provider="ollama", model="qwen2.5:7b"),
            full_store, registry, settings,
        )
        assert response.provider == "fakeprovider"
        assert response.model == "fake-model-v1"

    async def test_calls_the_llm_exactly_once(self, provider, registry, settings, full_store):
        await answer_question(
            QueryRequest(question="q"), full_store, registry, settings
        )
        assert provider.chat_calls == 1

    async def test_prompt_carries_the_context(self, provider, registry, settings, full_store):
        """Proves build_rag_messages() was actually used."""
        await answer_question(
            QueryRequest(question="How much parental leave?"), full_store, registry, settings
        )
        combined = "\n".join(m.content for m in provider.last_messages)
        assert "90 days of paid parental leave" in combined
        assert "How much parental leave?" in combined


class TestTopK:
    async def test_uses_the_settings_default(self, registry, settings, full_store):
        await answer_question(QueryRequest(question="q"), full_store, registry, settings)
        assert full_store.last_top_k == settings.rag_top_k

    async def test_request_overrides_the_default(self, registry, settings, full_store):
        await answer_question(
            QueryRequest(question="q", top_k=1), full_store, registry, settings
        )
        assert full_store.last_top_k == 1


class TestHistory:
    async def test_history_reaches_the_prompt(self, provider, registry, settings, full_store):
        request = QueryRequest(
            question="What about managers?",
            history=[
                {"role": "user", "content": "What is the notice period?"},
                {"role": "assistant", "content": "One month for employees."},
            ],
        )
        await answer_question(request, full_store, registry, settings)
        combined = "\n".join(m.content for m in provider.last_messages)
        assert "One month for employees." in combined

    async def test_no_history_is_fine(self, registry, settings, full_store):
        response = await answer_question(
            QueryRequest(question="q"), full_store, registry, settings
        )
        assert response.answer
