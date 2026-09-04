from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """One recommended model."""

    id: str
    provider: str
    label: str
    context: str
    #: Rough cost, or "free (local)". Cloud pricing moves - treat as a
    #: relative signal, not a quote.
    cost: str
    notes: str
    kind: str = "chat"  # "chat" | "embedding"


#: Local models, served by Ollama. Free, private, offline.
#: Sizes are the Q4 quantised download - roughly the VRAM needed too.
OLLAMA_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="llama3.2:3b",
        provider="ollama",
        label="Llama 3.2 3B",
        context="128K",
        cost="free (local)",
        notes="~2 GB. For machines with no dedicated GPU, or a tight VRAM budget.",
    ),
    ModelInfo(
        id="qwen2.5:7b",
        provider="ollama",
        label="Qwen 2.5 7B",
        context="32K",
        cost="free (local)",
        notes="~4.7 GB. This project's default - sized for an 8 GB VRAM budget.",
    ),
    ModelInfo(
        id="qwen2.5:14b",
        provider="ollama",
        label="Qwen 2.5 14B",
        context="32K",
        cost="free (local)",
        notes="~9 GB. Noticeably better reasoning; needs 12-16 GB VRAM.",
    ),
    ModelInfo(
        id="qwen2.5:32b",
        provider="ollama",
        label="Qwen 2.5 32B",
        context="32K",
        cost="free (local)",
        notes="~20 GB. Needs 24 GB+ VRAM.",
    ),
    ModelInfo(
        id="nomic-embed-text",
        provider="ollama",
        label="Nomic Embed Text",
        context="2K",
        cost="free (local)",
        notes="274 MB, 768 dimensions. The embedding model - never swap it "
        "without resetting the vector store.",
        kind="embedding",
    ),
]

#: Anthropic (Claude). Optional - needs ANTHROPIC_API_KEY.
#: NOTE: current Claude models reject `temperature` with a 400; the
#: provider strips it automatically. See anthropic_provider.py.
ANTHROPIC_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="claude-haiku-4-5",
        provider="anthropic",
        label="Claude Haiku 4.5",
        context="200K",
        cost="$1 / $5 per Mtok",
        notes="Fastest and cheapest. Good for simple, high-volume, latency-"
        "sensitive work.",
    ),
    ModelInfo(
        id="claude-sonnet-5",
        provider="anthropic",
        label="Claude Sonnet 5",
        context="1M",
        cost="$3 / $15 per Mtok",
        notes="Balanced. Near-Opus quality on coding and agentic work at "
        "lower cost.",
    ),
    ModelInfo(
        id="claude-opus-5",
        provider="anthropic",
        label="Claude Opus 5",
        context="1M",
        cost="$5 / $25 per Mtok",
        notes="The default here. Strongest for complex agentic and long-"
        "horizon work.",
    ),
    ModelInfo(
        id="claude-opus-4-8",
        provider="anthropic",
        label="Claude Opus 4.8",
        context="1M",
        cost="$5 / $25 per Mtok",
        notes="Previous-generation Opus. Still strong; the usual fallback "
        "target.",
    ),
    ModelInfo(
        id="claude-fable-5",
        provider="anthropic",
        label="Claude Fable 5",
        context="1M",
        cost="$10 / $50 per Mtok",
        notes="Most capable, and priciest. Only for the hardest reasoning "
        "tasks.",
    ),
]

#: OpenAI and Gemini catalogues are deliberately NOT hard-coded.
#: Their model ids change often enough that a list written today would be
#: wrong within months, and a wrong id fails as a confusing 404. Both
#: providers expose a listing endpoint, so `provider.list_models()`
#: fetches the real thing at runtime instead.
LIVE_ONLY_PROVIDERS = ("openai", "gemini")

#: provider name -> curated entries
RECOMMENDED: dict[str, list[ModelInfo]] = {
    "ollama": OLLAMA_MODELS,
    "anthropic": ANTHROPIC_MODELS,
}


def describe(provider: str, model_id: str) -> ModelInfo | None:
    """Look up curated info for a model id, or None if we have none.

    Matching ignores an Ollama ":latest" tag, because Ollama reports a
    model pulled without an explicit tag as "name:latest" while people
    write it as plain "name".
    """
    base = model_id.removesuffix(":latest")
    for info in RECOMMENDED.get(provider, []):
        if info.id == model_id or info.id == base:
            return info
    return None


def all_recommended() -> list[ModelInfo]:
    """Every curated model across providers, local first."""
    return [info for provider in ("ollama", "anthropic") for info in RECOMMENDED[provider]]
