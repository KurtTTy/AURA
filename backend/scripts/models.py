from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# Windows consoles default to cp1252, which cannot encode the box-drawing
# characters this script prints - and redirecting output to a file makes
# Python pick the locale encoding even when the console itself would cope.
# Force UTF-8 so `> out.txt` never crashes the run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from app.config import get_settings  # noqa: E402
from app.llm_providers import (  # noqa: E402
    Message,
    ProviderError,
    ProviderRegistry,
    all_recommended,
    describe,
)

CLOUD = {"anthropic", "openai", "gemini"}

#: How to turn each cloud provider on. Shown when nothing is configured.
SETUP_HINTS = {
    "anthropic": (
        "pip install anthropic   +   ANTHROPIC_API_KEY=... in .env",
        "https://console.anthropic.com/settings/keys",
    ),
    "openai": (
        "pip install openai      +   OPENAI_API_KEY=... in .env",
        "https://platform.openai.com/api-keys",
    ),
    "gemini": (
        "(no install needed)     +   GEMINI_API_KEY=... in .env",
        "https://aistudio.google.com/apikey",
    ),
}


def rule(text: str) -> None:
    print(f"\n{'=' * 72}\n  {text}\n{'=' * 72}")


async def show_available(registry: ProviderRegistry, settings) -> None:
    rule("AVAILABLE NOW")

    live = await registry.list_models()
    health = await registry.health()

    for name in registry.available:
        default_model = registry.default_model_for(name)
        status = "reachable" if health.get(name) else "UNREACHABLE"
        marker = "  <- default provider" if name == settings.default_provider else ""
        print(f"\n  {name.upper()}  ({status}){marker}")
        print(f"  default model: {default_model}")

        models = live.get(name, [])
        if not models:
            print("    (no models reported)")
            continue

        for model_id in models:
            info = describe(name, model_id)
            is_default = (
                model_id == default_model
                or model_id.removesuffix(":latest") == default_model
            )
            flag = " *" if is_default else "  "
            if info:
                print(f"   {flag} {model_id:<28} {info.context:>5}  {info.cost}")
                print(f"        {info.notes}")
            else:
                print(f"   {flag} {model_id}")

    print("\n  * = this provider's configured default")

    missing = sorted(CLOUD - set(registry.available))
    if missing:
        print("\n  Not configured (all optional - the system is fully local without them):")
        for name in missing:
            how, url = SETUP_HINTS[name]
            print(f"    {name:<10} {how}")
            print(f"    {'':<10} {url}")


def show_catalogue() -> None:
    rule("CURATED CATALOGUE (guidance, not a live list)")
    current = None
    for info in all_recommended():
        if info.provider != current:
            current = info.provider
            print(f"\n  {current.upper()}")
        kind = "" if info.kind == "chat" else f" [{info.kind}]"
        print(f"    {info.id:<24}{kind}")
        print(f"      {info.label} · {info.context} context · {info.cost}")
        print(f"      {info.notes}")

    print(
        "\n  OpenAI and Gemini are deliberately absent here: their model ids"
        "\n  change often, so a hard-coded list would go stale and a wrong id"
        "\n  fails as a confusing 404. Both are fetched live above instead."
    )


async def try_model(
    registry: ProviderRegistry, provider_name: str, model: str | None, prompt: str
) -> int:
    rule(f"TEST  provider={provider_name}  model={model or '(provider default)'}")
    print(f"  Prompt: {prompt!r}\n")

    try:
        provider = registry.get(provider_name)
        result = await provider.chat(
            [Message(role="user", content=prompt)], model=model, max_tokens=256
        )
    except ProviderError as exc:
        print(f"  FAILED: {exc}")
        return 1

    print(f"  ANSWER ({result.provider}/{result.model}):")
    print(f"  {result.text.strip()}\n")
    if result.prompt_tokens is not None:
        print(f"  tokens: {result.prompt_tokens} in / {result.completion_tokens} out")
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description="Browse and test available models.")
    parser.add_argument("--all", action="store_true", help="also show the curated catalogue")
    parser.add_argument("--provider", help="provider to test, e.g. ollama / anthropic")
    parser.add_argument("--model", help="model id to test (defaults to the provider's default)")
    parser.add_argument("--ask", default="Reply with exactly: OK", help="test prompt")
    args = parser.parse_args()

    settings = get_settings()
    registry = ProviderRegistry(settings)

    try:
        if args.provider:
            return await try_model(registry, args.provider, args.model, args.ask)

        await show_available(registry, settings)
        if args.all:
            show_catalogue()

        print(
            "\n  Use one for a single request without changing any config:"
            '\n    {"question": "...", "provider": "anthropic", "model": "claude-sonnet-5"}'
            "\n  Or change the standing default in .env (DEFAULT_PROVIDER / *_MODEL)."
        )
        return 0
    finally:
        await registry.aclose()


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
