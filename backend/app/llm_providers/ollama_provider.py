from __future__ import annotations

import json
from typing import Any, AsyncIterator, Sequence

import httpx

from .base import CompletionResult, LLMProvider, Message, ProviderError


class OllamaProvider(LLMProvider):
    """Talks to a local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        chat_model: str = "qwen2.5:7b",
        embed_model: str = "nomic-embed-text",
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model

        # One shared AsyncClient for the process. Creating a client per
        # request would throw away connection pooling and, on Windows,
        # can exhaust ephemeral ports under load. It's closed via
        # aclose() from the FastAPI shutdown hook.
        #
        # The timeout is split deliberately: connecting to a local
        # server should be near-instant (5s is already generous), but
        # *generating* 500 tokens on a 7B model can legitimately take
        # a couple of minutes, hence the long read timeout.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    # ── internals ────────────────────────────────────────────────

    def _options(self, temperature: float, max_tokens: int | None) -> dict[str, Any]:
        """Build Ollama's `options` block.

        Ollama names things after llama.cpp rather than after the OpenAI
        API, so the translation is worth spelling out:
            temperature -> temperature   (same)
            max_tokens  -> num_predict   (different name, same meaning)
        """
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        return options

    @staticmethod
    def _friendly_error(exc: Exception, model: str) -> ProviderError:
        """Turn low-level httpx failures into an actionable message.

        The two failures you will actually hit while building this are
        "Ollama isn't running" and "that model was never pulled", and
        the raw exceptions for both are unhelpful. Translate them here
        once instead of debugging them repeatedly later.
        """
        if isinstance(exc, httpx.ConnectError):
            return ProviderError(
                "Cannot reach Ollama. Is it running? Check with: ollama list"
            )
        if isinstance(exc, httpx.ReadTimeout):
            return ProviderError(
                f"Ollama timed out generating with '{model}'. The model may be "
                "too large for available VRAM and is swapping to CPU."
            )
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
            return ProviderError(
                f"Model '{model}' is not available locally. Pull it: ollama pull {model}"
            )
        return ProviderError(f"Ollama request failed: {exc}")

    # ── LLMProvider implementation ───────────────────────────────

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Non-streaming generation. See LLMProvider.chat for the contract."""
        use_model = model or self.chat_model
        payload = {
            "model": use_model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": self._options(temperature, max_tokens),
        }

        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise self._friendly_error(exc, use_model) from exc

        return CompletionResult(
            text=data.get("message", {}).get("content", ""),
            provider=self.name,
            model=use_model,
            # Ollama reports these as *_eval_count. prompt_eval_count is
            # how many tokens it read; eval_count is how many it wrote.
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            raw=data,
        )

    async def chat_stream(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Streaming generation, yielding text as the model produces it.

        With stream=True, Ollama responds with NDJSON: one JSON object
        per line, each carrying a fragment of the answer, terminated by
        an object with "done": true. We parse line by line and yield
        just the text so callers never see the wire format.
        """
        use_model = model or self.chat_model
        payload = {
            "model": use_model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "options": self._options(temperature, max_tokens),
        }

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        # A partial/garbled line is not worth killing the
                        # whole stream over; skip it and keep reading.
                        continue
                    fragment = event.get("message", {}).get("content", "")
                    if fragment:
                        yield fragment
                    if event.get("done"):
                        break
        except Exception as exc:
            raise self._friendly_error(exc, use_model) from exc

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Batch-embed texts with the local embedding model.

        /api/embed accepts a list under "input" and returns a matching
        list under "embeddings". Sending one request for many chunks is
        dramatically faster than one request per chunk, which is why the
        interface is batch-first.
        """
        if not texts:
            return []

        use_model = model or self.embed_model
        try:
            response = await self._client.post(
                "/api/embed",
                json={"model": use_model, "input": list(texts)},
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise self._friendly_error(exc, use_model) from exc

        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise ProviderError(
                f"Expected {len(texts)} embeddings from '{use_model}', "
                f"got {len(embeddings or [])}."
            )
        return embeddings

    async def health(self) -> bool:
        """Reachable AND has both configured models pulled.

        Checking only "is the server up" would let ingestion start and
        then fail hundreds of chunks in, so we verify the models exist
        too. Ollama reports names with tags ("qwen2.5:7b"), and a model
        pulled without an explicit tag lists as ":latest" - so we
        compare on the base name to avoid false negatives.
        """
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
            response.raise_for_status()
            available = {
                m.get("name", "").split(":")[0]
                for m in response.json().get("models", [])
            }
        except Exception:
            return False

        needed = {self.chat_model.split(":")[0], self.embed_model.split(":")[0]}
        return needed.issubset(available)

    async def list_models(self) -> list[str]:
        """Models pulled locally, i.e. what `ollama list` shows.

        Unlike the cloud providers, this is a list of what you have on
        disk rather than what the vendor offers - anything else needs an
        `ollama pull` first.
        """
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
            response.raise_for_status()
            return sorted(m.get("name", "") for m in response.json().get("models", []))
        except Exception:
            return []

    async def aclose(self) -> None:
        await self._client.aclose()
