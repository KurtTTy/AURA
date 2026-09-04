from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.db import init_db
from app.deps import get_data_session, get_registry, get_store
from app.llm_providers import ProviderError, ProviderRegistry, UnknownProviderError, describe
from app.models import (
    HealthResponse,
    Mode,
    ModelOption,
    ModelsResponse,
    ProviderModels,
    QueryRequest,
    QueryResponse,
)
from app.rag import VectorStore
from app.analyst import DataSession
from app.routers import analyst, chat, legal, rag
from app.routers.rag import answer_question

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown.

    Everything expensive is built once here and attached to app.state:
    the provider registry (HTTP connection pools) and the vector store
    (Chroma's on-disk index). Rebuilding these per request would make
    every call several hundred milliseconds slower.

    Code before `yield` runs at startup; after it, at shutdown.
    """
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    )

    settings.ensure_dirs()
    init_db()

    app.state.settings = settings
    app.state.registry = ProviderRegistry(settings)
    app.state.store = VectorStore(
        persist_dir=settings.vectorstore_dir,
        collection_name=settings.chroma_collection,
    )
    # One DuckDB session for the whole process, same reasoning as the
    # store: it holds an open connection plus every table loaded into it.
    app.state.data_session = DataSession(db_path=settings.analyst_db_path)

    # Check Ollama at startup rather than letting the first request
    # fail confusingly. A warning, not a crash - the API should still
    # come up so /health can explain what's wrong.
    if not await app.state.registry.get("ollama").health():
        logger.warning(
            "Ollama is not reachable, or '%s' / '%s' are not pulled. "
            "Fix with: ollama pull %s && ollama pull %s",
            settings.ollama_chat_model,
            settings.ollama_embed_model,
            settings.ollama_chat_model,
            settings.ollama_embed_model,
        )
    else:
        logger.info(
            "Ollama ready - chat=%s embed=%s",
            settings.ollama_chat_model,
            settings.ollama_embed_model,
        )

    logger.info("Vector store '%s': %d chunks indexed", settings.chroma_collection, app.state.store.count())

    yield

    await app.state.registry.aclose()
    app.state.data_session.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="AURA",
    description="A local AI assistant. Three modes: chat, documents, data.",
    version="0.1.0",
    lifespan=lifespan,
)

# A browser client served from another localhost port counts as a
# different origin, so it needs CORS permission. Localhost only - do not
# widen this to "*" if the API is ever exposed beyond your own machine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(rag.router)
app.include_router(analyst.router)
app.include_router(legal.router)


# ═══════════════════════════════════════════════════════════════════
#  Mode routing
# ═══════════════════════════════════════════════════════════════════

#: Words that suggest a tabular-analysis request rather than a document
#: lookup. Crude on purpose - see resolve_mode().
_ANALYST_HINTS = (
    "average", "mean", "median", "sum", "total", "correlation",
    "trend", "plot", "chart", "graph", "distribution", "per month",
    "group by", "top 10", "outlier",
)

_LEGAL_HINTS = (
    "clause", "liability", "gdpr", "compliance", "contract",
    "terms of service", "privacy policy", "regulation", "ai ethics",
    "indemnity",
)


def resolve_mode(request: QueryRequest, store: VectorStore) -> Mode:
    """Decide which persona should handle a request when mode="auto".

    Intentionally a simple keyword heuristic, not an LLM classifier.
    Two reasons: it's instant and free, and it's debuggable - when
    routing goes wrong you can read this function and see exactly why.

    A common upgrade later is to ask the LLM to classify intent. Worth
    doing only if this proves inadequate in practice; it adds a full
    model round-trip to every single request.

    Precedence: explicit mode > legal > analyst > rag > chat.
    """
    if request.mode is not Mode.AUTO:
        return request.mode

    question = request.question.lower()

    if any(hint in question for hint in _LEGAL_HINTS):
        return Mode.LEGAL
    if any(hint in question for hint in _ANALYST_HINTS):
        return Mode.ANALYST

    # Only route to RAG if documents actually exist. Otherwise RAG would
    # return "no documents indexed" for a question plain chat could have
    # answered fine.
    if store.count() > 0:
        return Mode.RAG

    return Mode.CHAT


@app.post("/api/query", response_model=QueryResponse, tags=["router"])
async def unified_query(
    request: QueryRequest,
    store: VectorStore = Depends(get_store),
    session: DataSession = Depends(get_data_session),
    registry: ProviderRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    """Single entry point for every mode.

    Set `mode` explicitly to force one, or leave it as "auto" and let
    resolve_mode() choose.
    """
    mode = resolve_mode(request, store)
    logger.info("Routing %r -> mode=%s", request.question[:60], mode.value)

    try:
        if mode is Mode.RAG:
            return await answer_question(request, store, registry, settings)

        if mode is Mode.CHAT:
            provider = registry.get(request.provider)
            result = await provider.chat(
                chat.build_chat_messages(request), model=request.model
            )
            return QueryResponse(
                answer=result.text,
                mode=Mode.CHAT,
                provider=result.provider,
                model=result.model,
            )

        if mode is Mode.ANALYST:
            return await analyst.analyze(request, session, registry, settings)

        # The legal model is not built; the route returns 501.
        raise HTTPException(
            status_code=501,
            detail=(
                f"Mode '{mode.value}' is not implemented yet. "
                "Send mode='chat', 'rag' or 'analyst' explicitly to override routing."
            ),
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except UnknownProviderError as exc:
        # A bad provider name is the caller's mistake, not an outage -
        # retrying changes nothing, so 503 would be misleading advice.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ═══════════════════════════════════════════════════════════════════
#  Model discovery
# ═══════════════════════════════════════════════════════════════════

#: Providers that only exist when an API key is configured.
_CLOUD_PROVIDERS = {"anthropic", "openai", "gemini"}


@app.get("/api/models", response_model=ModelsResponse, tags=["system"])
async def list_models(
    registry: ProviderRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> ModelsResponse:
    """Every model you can currently pick, grouped by provider.

    Model ids are fetched live from each provider's own listing endpoint
    rather than hard-coded, so this can't go stale as vendors add and
    retire models. The curated catalogue then annotates whatever it
    recognises with context size, cost, and guidance.

    Any `id` returned here is valid in the `model` field of a request,
    paired with its `provider`:

        {"question": "...", "provider": "anthropic", "model": "claude-sonnet-5"}

    `scripts/models.py` prints the same data in the terminal.
    """
    live = await registry.list_models()
    health = await registry.health()

    providers: list[ProviderModels] = []
    for name in registry.available:
        default_model = registry.default_model_for(name)
        options: list[ModelOption] = []

        for model_id in live.get(name, []):
            info = describe(name, model_id)
            options.append(
                ModelOption(
                    id=model_id,
                    provider=name,
                    label=info.label if info else None,
                    context=info.context if info else None,
                    cost=info.cost if info else None,
                    notes=info.notes if info else None,
                    kind=info.kind if info else "chat",
                    # Ollama reports an untagged model as "name:latest",
                    # so compare on the base name too.
                    is_default=model_id == default_model
                    or model_id.removesuffix(":latest") == default_model,
                )
            )

        providers.append(
            ProviderModels(
                provider=name,
                available=health.get(name, False),
                is_default_provider=name == settings.default_provider,
                default_model=default_model,
                requires_api_key=name in _CLOUD_PROVIDERS,
                models=options,
            )
        )

    return ModelsResponse(default_provider=settings.default_provider, providers=providers)


# ═══════════════════════════════════════════════════════════════════
#  Health
# ═══════════════════════════════════════════════════════════════════


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health(
    store: VectorStore = Depends(get_store),
    registry: ProviderRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Is the system ready to answer questions?

    Check this first whenever something misbehaves. It tells you
    whether Ollama is up with the right models pulled, and how many
    chunks are indexed - which between them explain most failures.
    """
    provider_health = await registry.health()
    return HealthResponse(
        status="ok" if provider_health.get("ollama") else "degraded",
        providers=provider_health,
        chat_model=settings.ollama_chat_model,
        embed_model=settings.ollama_embed_model,
        collection=settings.chroma_collection,
        indexed_chunks=store.count(),
    )


@app.get("/", tags=["system"])
async def root() -> dict:
    return {
        "name": "AURA",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
