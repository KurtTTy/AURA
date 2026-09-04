from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Sequence

import httpx

from .base import CompletionResult, LLMProvider, Message, ProviderError

logger = logging.getLogger(__name__)

_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(LLMProvider):
    """Google Gemini via the AI Studio REST API (free tier)."""

    name = "gemini"

    #: Google renamed the grounding tool between API generations. Newer
    #: models take `google_search`; Gemini 1.5-era ones took
    #: `google_search_retrieval`. We try the modern one and fall back
    #: once on a 400, rather than making you guess which era your model
    #: belongs to.
    _SEARCH_TOOL_SHAPES = ({"google_search": {}}, {"google_search_retrieval": {}})

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        timeout: float = 60.0,
        enable_search: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("GeminiProvider requires an API key")
        self._api_key = api_key
        self.model = model
        self._client = httpx.AsyncClient(base_url=_API_ROOT, timeout=timeout)

        #: When true, requests ask Google to search the web first. This is
        #: the ONLY route to live information anywhere in this project.
        self.search_enabled = enable_search
        #: Index into _SEARCH_TOOL_SHAPES - which spelling worked. Sticky
        #: after the first success so we don't re-pay the failed attempt.
        self._tool_shape = 0

    # ── internals ────────────────────────────────────────────────

    @staticmethod
    def _to_gemini_payload(
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Translate our Message list into Gemini's request shape.

        Gemini differs from the OpenAI/Ollama convention in two ways,
        both handled here so the rest of the codebase never has to care:

          1. The assistant role is called "model", not "assistant".
          2. System prompts are NOT a message in the list; they go in a
             separate top-level `systemInstruction` field.

        This translation is exactly the work the provider abstraction
        exists to contain.
        """
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        for message in messages:
            if message.role == "system":
                system_parts.append(message.content)
                continue
            contents.append(
                {
                    "role": "model" if message.role == "assistant" else "user",
                    "parts": [{"text": message.content}],
                }
            )

        generation_config: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }
        return payload

    def _scrub(self, text: str) -> str:
        """Remove the API key from anything we're about to show or log.

        Gemini takes the key as a URL query parameter, so httpx's error
        messages embed the whole URL - key included. That string then ends
        up in terminal output, log files, and pasted bug reports. Any
        provider that authenticates via the URL needs this; header-based
        ones (Anthropic, OpenAI) don't leak the same way.
        """
        return text.replace(self._api_key, "***REDACTED***") if self._api_key else text

    def _wrap_error(self, exc: Exception) -> ProviderError:
        """Turn an httpx failure into something readable, key removed."""
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code == 429:
                return ProviderError(
                    "Gemini rate limit hit (429). Free-tier quotas are per-model "
                    "and per-minute, and grounded requests count heavier. Wait a "
                    "minute, or try a lighter model (/models to see them)."
                )
            if code in (401, 403):
                return ProviderError(
                    f"Gemini rejected the key ({code}). Check GEMINI_API_KEY in .env."
                )
            if code == 404:
                return ProviderError(
                    "Gemini does not recognise that model. /models lists what "
                    "your key can actually reach."
                )
            return ProviderError(self._scrub(f"Gemini API error {code}: {exc}"))
        return ProviderError(self._scrub(f"Gemini request failed: {exc}"))

    @staticmethod
    def _grounding_sources(data: dict[str, Any]) -> list[str]:
        """Pull the web pages Google actually consulted.

        Lives under candidates[].groundingMetadata.groundingChunks[].web,
        each with a `uri` and usually a `title`. Written defensively
        because the block is absent entirely when no search happened -
        which is most requests.
        """
        found: list[str] = []
        for candidate in data.get("candidates") or []:
            metadata = candidate.get("groundingMetadata") or {}
            for chunk in metadata.get("groundingChunks") or []:
                web = chunk.get("web") or {}
                uri, title = web.get("uri"), web.get("title")
                if uri:
                    found.append(f"{title} — {uri}" if title else uri)

        # Dedupe while keeping Google's ordering (most relevant first).
        seen: set[str] = set()
        return [s for s in found if not (s in seen or seen.add(s))]

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with a one-shot retry on the other grounding tool name.

        A 400 when search is on almost always means this model wants the
        other spelling. Retrying once is cheaper than making the user
        diagnose an API-version mismatch from a raw error string.
        """
        response = await self._client.post(
            path, params={"key": self._api_key}, json=payload
        )
        if (
            response.status_code == 400
            and self.search_enabled
            and self._tool_shape + 1 < len(self._SEARCH_TOOL_SHAPES)
        ):
            self._tool_shape += 1
            logger.warning(
                "Gemini rejected the grounding tool; retrying with %s",
                next(iter(self._SEARCH_TOOL_SHAPES[self._tool_shape])),
            )
            payload = {**payload, "tools": [self._SEARCH_TOOL_SHAPES[self._tool_shape]]}
            response = await self._client.post(
                path, params={"key": self._api_key}, json=payload
            )

        response.raise_for_status()
        return response.json()

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Pull the answer text out of Gemini's nested response.

        Shape: candidates[0].content.parts[*].text - written defensively
        because a response blocked by safety filters has candidates but
        no parts, and we want "" rather than a KeyError.
        """
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts") or []
        return "".join(part.get("text", "") for part in parts)

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
        payload = self._to_gemini_payload(messages, temperature, max_tokens)
        if self.search_enabled:
            payload["tools"] = [self._SEARCH_TOOL_SHAPES[self._tool_shape]]

        try:
            data = await self._post(f"/models/{use_model}:generateContent", payload)
        except Exception as exc:
            raise self._wrap_error(exc) from exc

        usage = data.get("usageMetadata", {})
        return CompletionResult(
            text=self._extract_text(data),
            provider=self.name,
            model=use_model,
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
            sources=self._grounding_sources(data),
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
        """Stream via server-sent events (alt=sse).

        Gemini's SSE format prefixes each event with "data: ", unlike
        Ollama's bare NDJSON - another provider difference absorbed here.
        """
        use_model = model or self.model
        payload = self._to_gemini_payload(messages, temperature, max_tokens)
        if self.search_enabled:
            # Grounding works while streaming, but the source list arrives
            # in later chunks and a generator has nowhere to hand it back.
            # Callers that need citations should use chat() instead - see
            # the note in ask.py.
            payload["tools"] = [self._SEARCH_TOOL_SHAPES[self._tool_shape]]

        try:
            async with self._client.stream(
                "POST",
                f"/models/{use_model}:streamGenerateContent",
                params={"key": self._api_key, "alt": "sse"},
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    body = line[len("data:") :].strip()
                    if not body or body == "[DONE]":
                        continue
                    try:
                        fragment = self._extract_text(json.loads(body))
                    except json.JSONDecodeError:
                        continue
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
        """Always refuses. See the module docstring for the reasoning."""
        raise NotImplementedError(
            "Embeddings must always be produced locally so every vector in "
            "the store shares one coordinate space. Use the Ollama provider."
        )

    async def health(self) -> bool:
        try:
            response = await self._client.get(
                "/models", params={"key": self._api_key}, timeout=5.0
            )
            return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """Models this key can use, fetched live.

        Gemini returns names as "models/gemini-x" while requests take the
        bare id, so the prefix is stripped here. Filtered to models that
        actually support generateContent - the raw list also includes
        embedding-only models we must never use (see embed() above).
        """
        try:
            response = await self._client.get(
                "/models", params={"key": self._api_key, "pageSize": 100}, timeout=10.0
            )
            response.raise_for_status()
            models = response.json().get("models", [])
        except Exception:
            return []

        return sorted(
            m["name"].removeprefix("models/")
            for m in models
            if "generateContent" in (m.get("supportedGenerationMethods") or [])
        )

    async def aclose(self) -> None:
        await self._client.aclose()
