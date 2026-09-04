from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.llm_providers import ProviderRegistry
from app.models import SourceChunk
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned by similarity search.

    Distinct from rag.chunking.Chunk: that one is a piece produced
    during ingestion; this one is a piece found during a query, and so
    carries a relevance score and the id it was stored under.
    """

    text: str
    source: str
    chunk_index: int
    score: float | None
    metadata: dict[str, Any]

    def to_schema(self) -> SourceChunk:
        """Convert to the Pydantic model returned over HTTP.

        Kept as an explicit method so the internal type can evolve
        without changing the public API shape.
        """
        return SourceChunk(
            text=self.text,
            source=self.source,
            chunk_index=self.chunk_index,
            score=self.score,
            metadata={
                k: v
                for k, v in self.metadata.items()
                if isinstance(v, (str, int, float, bool))
            },
        )


async def embed_query(query: str, registry: ProviderRegistry) -> list[float]:
    """Embed a single query string.

    Always uses registry.embedding_provider() - the local Ollama model -
    so query vectors stay comparable to stored vectors regardless of
    which provider will generate the final answer.
    """
    vectors = await registry.embedding_provider().embed([query])
    return vectors[0]


async def retrieve(
    query: str,
    store: VectorStore,
    registry: ProviderRegistry,
    top_k: int = 5,
    where: dict[str, Any] | None = None,
    min_score: float | None = None,
) -> list[RetrievedChunk]:
    """Find the top_k chunks most similar to `query`.

    Args:
        query: The user's question, as written.
        store: The vector store to search.
        registry: Provides the embedding model.
        top_k: Maximum chunks to return.
        where: Optional metadata filter, e.g. {"source": "policy.pdf"}.
        min_score: Drop results scoring below this. Scores are
            1 - cosine_distance, so roughly: >0.5 clearly related,
            0.3-0.5 loosely related, <0.3 usually noise.

            Leave as None until you've seen real scores from your own
            documents - a threshold set blind will silently discard
            good results. Tune it by looking at the `sources` array in
            actual API responses.

    Returns:
        Chunks sorted most-relevant first. Empty list if the store is
        empty or nothing clears min_score.
    """
    if store.count() == 0:
        logger.warning("Vector store '%s' is empty - ingest documents first.", store.collection_name)
        return []

    embedding = await embed_query(query, registry)
    hits = store.query(embedding, top_k=top_k, where=where)

    chunks: list[RetrievedChunk] = []
    for hit in hits:
        score = hit.get("score")
        if min_score is not None and score is not None and score < min_score:
            continue
        metadata = hit.get("metadata") or {}
        chunks.append(
            RetrievedChunk(
                text=hit.get("text", ""),
                source=str(metadata.get("source", "unknown")),
                chunk_index=int(metadata.get("chunk_index", 0)),
                score=score,
                metadata=dict(metadata),
            )
        )

    logger.info(
        "Retrieved %d chunk(s) for query %r (top score: %s)",
        len(chunks),
        query[:60],
        f"{chunks[0].score:.3f}" if chunks and chunks[0].score is not None else "n/a",
    )
    return chunks
