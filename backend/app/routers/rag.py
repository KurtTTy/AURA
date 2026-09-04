from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.deps import get_registry, get_store
from app.llm_providers import Message, ProviderError, ProviderRegistry, UnknownProviderError
from app.models import (
    IngestRequest,
    IngestResponse,
    Mode,
    QueryRequest,
    QueryResponse,
)
from app.rag import (
    NO_CONTEXT_ANSWER,
    SUPPORTED_EXTENSIONS,
    VectorStore,
    build_rag_messages,
    ingest_paths,
    retrieve,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag", tags=["rag"])


async def answer_question( request: QueryRequest, store: VectorStore, registry: ProviderRegistry, settings: Settings,) -> QueryResponse:
    """Retrieve relevant chunks, prompt the LLM, return a grounded answer.

    The seam where retrieval, prompting, and generation meet.

    Retrieval happens BEFORE prompting, because the chunks are prompt
    inputs. When retrieval returns nothing the LLM is never called at
    all - a 7B model handed no context will confabulate fluently - so
    NO_CONTEXT_ANSWER is returned directly.

    Args:
        request: `question`, `history`, `provider`, `model`, `top_k`.
        store: Vector store to search.
        registry: Source of both the embedding and generation providers.
        settings: Supplies rag_top_k when request.top_k is None.

    Returns:
        QueryResponse with the answer, its sources, and the provider and
        model that ACTUALLY ran - which can differ from what was asked
        for, so both are read back off the CompletionResult.

    Raises:
        ProviderError: propagated; the route handler turns it into a 503.
    """
    top_k = request.top_k or settings.rag_top_k
    chunks = await retrieve(request.question, store, registry, top_k=top_k)
    provider = registry.get(request.provider)
    model_name = request.model or registry.default_model_for(provider.name)
    if not chunks:
        return QueryResponse(
            answer=NO_CONTEXT_ANSWER,
            sources=[],
            mode=Mode.RAG,
            provider=provider.name,
            model=model_name,
        )
    history = [Message(role=m.role, content=m.content) for m in request.history]
    messages = build_rag_messages(request.question, chunks, history)
    result = await provider.chat(messages, model=model_name)
    return QueryResponse(
        answer=result.text,
        sources=[c.to_schema() for c in chunks],
        mode=Mode.RAG,
        provider=result.provider,
        model=result.model,
    )
    


# ═══════════════════════════════════════════════════════════════════
#  Routes below are scaffolded and complete.
# ═══════════════════════════════════════════════════════════════════


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    store: VectorStore = Depends(get_store),
    registry: ProviderRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    """Ask a question answered from your ingested documents."""
    try:
        return await answer_question(request, store, registry, settings)
    except NotImplementedError as exc:
        # 501 Not Implemented is the honest status while the glue is
        # unwritten - clearer than a 500 that looks like a real bug.
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except UnknownProviderError as exc:
        # A bad provider name is the caller's mistake, not an outage -
        # retrying changes nothing, so 503 would be misleading advice.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderError as exc:
        # 503 Service Unavailable: our code is fine, the LLM backend
        # isn't reachable (Ollama stopped, model not pulled, timeout).
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/ingest", response_model=IngestResponse)
async def ingest_from_paths(
    request: IngestRequest,
    store: VectorStore = Depends(get_store),
    registry: ProviderRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """Ingest files or directories already on disk.

    Relative paths resolve against data/raw/, so {"paths": ["notes"]}
    ingests data/raw/notes/.
    """
    resolved: list[Path] = []
    for raw_path in request.paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = settings.raw_dir / path
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {path}")
        resolved.append(path)

    try:
        return await ingest_paths(
            resolved,
            store,
            registry,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            recursive=request.recursive,
            replace_existing=request.replace_existing,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except UnknownProviderError as exc:
        # A bad provider name is the caller's mistake, not an outage -
        # retrying changes nothing, so 503 would be misleading advice.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/upload", response_model=IngestResponse)
async def upload_and_ingest(
    files: list[UploadFile] = File(...),
    store: VectorStore = Depends(get_store),
    registry: ProviderRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """Upload documents and ingest them immediately.

    The HTTP entry point for document questions.
    Uploads are saved to data/raw/ first so they can be re-ingested
    later (after a chunking change) without re-uploading.
    """
    settings.ensure_dirs()
    saved: list[Path] = []

    for upload in files:
        # Path(...).name strips any directory components a client might
        # send, so "../../etc/passwd" becomes "passwd". Never trust an
        # uploaded filename as a path.
        filename = Path(upload.filename or "unnamed").name
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type: {filename}. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                ),
            )

        destination = settings.raw_dir / filename
        with destination.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        saved.append(destination)
        logger.info("Saved upload: %s", destination)

    try:
        return await ingest_paths(
            saved,
            store,
            registry,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except UnknownProviderError as exc:
        # A bad provider name is the caller's mistake, not an outage -
        # retrying changes nothing, so 503 would be misleading advice.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/documents")
async def list_documents(store: VectorStore = Depends(get_store)) -> dict:
    """List indexed documents and total chunk count."""
    return {
        "collection": store.collection_name,
        "sources": store.sources(),
        "total_chunks": store.count(),
    }


@router.delete("/documents/{source}")
async def delete_document(source: str, store: VectorStore = Depends(get_store)) -> dict:
    """Remove every chunk belonging to one source file."""
    store.delete_by_source(source)
    return {"deleted": source, "remaining_chunks": store.count()}


@router.post("/reset")
async def reset_collection(store: VectorStore = Depends(get_store)) -> dict:
    """Wipe the entire vector store.

    Run this after changing the embedding model or your chunking
    strategy - existing vectors are not comparable to new ones.
    """
    store.reset()
    return {"status": "reset", "collection": store.collection_name, "total_chunks": 0}
