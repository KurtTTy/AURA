from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from app.db import Document, session_scope
from app.llm_providers import ProviderRegistry
from app.models import IngestedDocument, IngestResponse
from .chunking import Chunk, chunk_text, make_chunk_id
from .loaders import (
    UnsupportedFileType,
    discover_files,
    file_hash,
    load_document,
)
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)

#: Texts sent to Ollama per embedding request. Batching is a large speed
#: win over one-call-per-chunk, but an unbounded batch can exhaust VRAM
#: on a 16GB card once the chat model is also loaded. 32 is a safe
#: middle ground for nomic-embed-text; raise it if ingestion feels slow
#: and you have headroom.
EMBED_BATCH_SIZE = 32


async def _embed_in_batches(
    texts: Sequence[str],
    registry: ProviderRegistry,
) -> list[list[float]]:
    """Embed many texts in fixed-size batches, preserving order.

    Order preservation matters: embeddings[i] must correspond to
    texts[i] when they're handed to the vector store together.
    """
    provider = registry.embedding_provider()
    vectors: list[list[float]] = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = list(texts[start : start + EMBED_BATCH_SIZE])
        vectors.extend(await provider.embed(batch))
        logger.debug("Embedded %d/%d chunks", len(vectors), len(texts))

    return vectors


async def ingest_file(
    path: Path,
    store: VectorStore,
    registry: ProviderRegistry,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    replace_existing: bool = True,
) -> IngestedDocument:
    """Run one file through the full pipeline.

    Args:
        path: File to ingest.
        store: Destination vector store.
        registry: Supplies the embedding model.
        chunk_size / chunk_overlap: Passed through to chunk_text().
        replace_existing: Delete this source's previous chunks first.
            Keep True when re-ingesting an edited document, otherwise
            stale chunks from the old version remain searchable.

    Returns:
        A per-document report. `skipped_reason` is set (and chunks=0)
        when nothing was indexed, rather than raising - one unreadable
        file shouldn't abort a whole directory ingest.
    """
    source = path.name

    try:
        text = load_document(path)
    except (UnsupportedFileType, FileNotFoundError) as exc:
        return IngestedDocument(source=source, chunks=0, characters=0, skipped_reason=str(exc))

    if not text.strip():
        return IngestedDocument(
            source=source,
            chunks=0,
            characters=0,
            skipped_reason="No extractable text (a scanned PDF would need OCR).",
        )

    # ── chunk ────────────────────────────────────────────────────
    chunks: list[Chunk] = chunk_text(
        text,
        source=source,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        extra_metadata={"file_type": path.suffix.lower().lstrip(".")},
    )
    if not chunks:
        return IngestedDocument(
            source=source, chunks=0, characters=len(text), skipped_reason="Produced no chunks."
        )

    # ── embed ────────────────────────────────────────────────────
    embeddings = await _embed_in_batches([c.text for c in chunks], registry)

    # ── store ────────────────────────────────────────────────────
    if replace_existing:
        store.delete_by_source(source)

    ids = [make_chunk_id(source, c.index, c.text) for c in chunks]
    metadatas: list[dict[str, Any]] = []
    for chunk in chunks:
        # Chroma only accepts scalar metadata values, so anything
        # non-scalar a chunker attached is stringified rather than
        # blowing up the insert.
        metadata = {
            k: (v if isinstance(v, (str, int, float, bool)) else str(v))
            for k, v in chunk.metadata.items()
        }
        metadata.setdefault("source", source)
        metadata["chunk_index"] = chunk.index
        metadatas.append(metadata)

    store.add(
        ids=ids,
        texts=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # ── record metadata ──────────────────────────────────────────
    with session_scope() as db:
        record = db.query(Document).filter_by(source=source).one_or_none()
        if record is None:
            record = Document(source=source)
            db.add(record)
        record.file_type = path.suffix.lower().lstrip(".")
        record.chunk_count = len(chunks)
        record.char_count = len(text)
        record.content_hash = file_hash(path)

    logger.info("Ingested %s -> %d chunks (%d chars)", source, len(chunks), len(text))
    return IngestedDocument(source=source, chunks=len(chunks), characters=len(text))


async def ingest_paths(
    paths: Sequence[Path],
    store: VectorStore,
    registry: ProviderRegistry,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    recursive: bool = True,
    replace_existing: bool = True,
) -> IngestResponse:
    """Ingest any mix of files and directories.

    Directories are expanded to their supported files. Each file is
    processed independently so one failure doesn't stop the rest.
    """
    discovered: list[Path] = []
    for path in paths:
        discovered.extend(discover_files(path, recursive=recursive))

    # Dedupe while keeping order - the same file can appear via both an
    # explicit path and its parent directory.
    seen: set[Path] = set()
    unique = [p for p in discovered if not (p in seen or seen.add(p))]

    reports: list[IngestedDocument] = []
    for path in unique:
        try:
            reports.append(
                await ingest_file(
                    path,
                    store,
                    registry,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    replace_existing=replace_existing,
                )
            )
        except NotImplementedError:
            # chunk_text() isn't written yet - surface it clearly rather
            # than burying it as a per-file "skipped" line.
            raise
        except Exception as exc:
            logger.exception("Failed to ingest %s", path)
            reports.append(
                IngestedDocument(
                    source=path.name, chunks=0, characters=0, skipped_reason=str(exc)
                )
            )

    return IngestResponse(
        documents=reports,
        total_chunks=sum(r.chunks for r in reports),
        collection=store.collection_name,
    )
