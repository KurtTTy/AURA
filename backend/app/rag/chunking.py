from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Chunk:
    """One retrievable piece of a document.

    `metadata` always includes "source". Chroma restricts its values to
    str/int/float/bool - no nested dicts or lists.
    """

    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def make_chunk_id(source: str, index: int, text: str) -> str:
    """Stable id: same source + position + content gives the same id, so
    re-ingesting an unchanged document upserts instead of duplicating.

    Hashing the text means an edited paragraph gets a new id, which is
    correct - it is different content now.
    """
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{source}::{index}::{digest}"


def chunk_text(text: str, *, source: str, chunk_size: int = 1000, chunk_overlap: int = 150, 
               extra_metadata: dict[str, Any] | None = None,) -> list[Chunk]:

    """Split a document's text into overlapping chunks.

    Args:
        text: Full document text from loaders.load_document().
        source: Ends up in every chunk's metadata under "source";
            delete_by_source() and the citations both depend on it.
        chunk_size: Soft ceiling - a chunk may end early at a clean
            boundary rather than slicing mid-sentence.
        chunk_overlap: Characters repeated between chunks. < chunk_size.
        extra_metadata: Merged into every chunk's metadata.

    Returns:
        Chunks in order, `index` from 0 with no gaps. Empty if no content.

    Raises:
        ValueError: overlap >= size would make no forward progress.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []
    
    base_metadata = {"source": source}
    if extra_metadata:
        base_metadata.update(extra_metadata)
        base_metadata["source"] = source  # Ensure source is always present

    chunks: list[Chunk] = []
    seen_text: set[str] = set()
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            search_floor = start + int(chunk_size * 0.7)
            for separator in ("\n\n", "\n", ". ", " "):
                boundary = text.rfind(separator, search_floor, end)
                if boundary != -1:
                    end = boundary + len(separator)
                    break

        piece = text[start:end].strip()
        if piece and piece not in seen_text:
            chunks.append(Chunk(text=piece, index=index, metadata=dict(base_metadata)))
            seen_text.add(piece)
            index += 1

        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks
        

