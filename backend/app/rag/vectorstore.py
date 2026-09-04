from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin wrapper over a persistent Chroma collection - swapping for
    Qdrant or pgvector would change only this file."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "documents",
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # hnsw:space="cosine" sets the distance metric. Chroma's default
        # is squared L2, but text embeddings are conventionally compared
        # by cosine similarity - direction matters, magnitude doesn't.
        #
        # This is fixed when the collection is CREATED and cannot be
        # changed later. To switch metrics you must delete and re-index.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection_name = collection_name

    # ── writes ───────────────────────────────────────────────────

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        """Insert or overwrite chunks.

        upsert, not add - re-ingesting the same document replaces its rows
        instead of erroring, so ingestion is repeatable.

        All four sequences must be equal length and index-aligned.
        """
        if not ids:
            return
        lengths = {len(ids), len(texts), len(embeddings), len(metadatas)}
        if len(lengths) != 1:
            raise ValueError(
                f"add() requires equal-length sequences, got lengths {lengths}"
            )

        self._collection.upsert(
            ids=list(ids),
            documents=list(texts),
            embeddings=[list(v) for v in embeddings],
            metadatas=list(metadatas),
        )

    def delete_by_source(self, source: str) -> None:
        """Remove every chunk from one file.

        Needed before re-ingesting an edited document: if it now produces
        fewer chunks, the leftovers would still be retrieved.
        """
        self._collection.delete(where={"source": source})

    def reset(self) -> None:
        """Delete the collection and recreate it empty. Required after
        changing the embedding model or chunking strategy."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    # ── reads ────────────────────────────────────────────────────

    def query(
        self,
        embedding: Sequence[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Find the top_k chunks nearest to a query embedding.

        `embedding` must come from the SAME model that embedded the stored
        chunks. `where` filters on metadata, e.g. {"source": "policy.pdf"}.

        Returns dicts with id, text, metadata, distance, score - nearest
        first. distance is raw cosine (0 = identical); score is 1-distance,
        flipped so higher means more relevant.
        """
        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns one list per query embedding; we sent one, so
        # everything of interest is at index 0. A [[]] means no matches.
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            distance = distances[index] if index < len(distances) else None
            hits.append(
                {
                    "id": chunk_id,
                    "text": documents[index] if index < len(documents) else "",
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "distance": distance,
                    "score": None if distance is None else 1.0 - float(distance),
                }
            )
        return hits

    def count(self) -> int:
        """Total chunks currently indexed."""
        return self._collection.count()

    def sources(self) -> list[str]:
        """Distinct source filenames present in the collection.

        Chroma has no DISTINCT, so this pulls metadata and dedupes in
        Python. Fine at personal-knowledge-base scale.
        """
        result = self._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []
        return sorted({str(m.get("source", "")) for m in metadatas if m} - {""})
