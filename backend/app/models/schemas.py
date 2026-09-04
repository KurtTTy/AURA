from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Mode(str, Enum):
    """The four personas this one system presents, plus auto-routing.

    Inheriting from `str` means these serialise as plain strings in JSON
    ("rag", not an object) while still being a real enum in Python.
    """

    CHAT = "chat"       # plain conversation, no retrieval
    RAG = "rag"         # grounded in your ingested documents
    ANALYST = "analyst"  # queries tabular data with SQL
    LEGAL = "legal"     # reserved; the endpoint returns 501
    AUTO = "auto"       # let the mode router decide


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    """One request into the unified /api/query endpoint."""

    question: str = Field(min_length=1, description="The user's question.")
    mode: Mode = Mode.AUTO
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Prior turns, oldest first. Empty for a fresh conversation.",
    )
    # Per-request provider/model override. This is the payoff of the
    # provider abstraction: the caller can say "answer this one with
    # Gemini" without any code change.
    provider: str | None = None
    model: str | None = None
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Chunks to retrieve (RAG mode). None = server default.",
    )
    stream: bool = False


class SourceChunk(BaseModel):
    """One retrieved chunk, returned alongside an answer.

    Returning sources is what separates a RAG system from a chatbot that
    sounds confident. Every grounded answer must be traceable back to
    the text it came from.
    """

    text: str
    source: str = Field(description="Originating filename or path.")
    chunk_index: int = Field(description="Position of this chunk within its document.")
    score: float | None = Field(
        default=None,
        description="Similarity score. Higher = more relevant to the query.",
    )
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    mode: Mode = Field(description="The mode that actually handled the request.")
    provider: str
    model: str
    sources: list[SourceChunk] = Field(
        default_factory=list,
        description="Empty for non-RAG modes.",
    )

    # ── Analyst-only ─────────────────────────────────────────────
    # All optional with defaults, so every existing response still
    # validates. Adding a REQUIRED field here would break every caller
    # and every existing test at once.
    sql: list[str] = Field(
        default_factory=list,
        description=(
            "Queries the analyst ran, in order. Empty for other modes. "
            "This is how a human checks the model's arithmetic - an "
            "analyst answer without its queries is just a claim."
        ),
    )
    chart_path: str | None = Field(
        default=None,
        description="Chart file written by the analyst, if any. A str, not a Path, because JSON has no path type.",
    )
    turns: int | None = Field(
        default=None,
        description="Model calls the analyst loop used. None for other modes.",
    )
    exports: list[str] = Field(
        default_factory=list,
        description="Spreadsheet files the analyst wrote for the user to open.",
    )


class IngestRequest(BaseModel):
    """Ingest documents already sitting on disk (as opposed to uploaded)."""

    paths: list[str] = Field(
        min_length=1,
        description="Files or directories to ingest, relative to data/raw or absolute.",
    )
    recursive: bool = True
    replace_existing: bool = Field(
        default=True,
        description="Delete a document's previous chunks before re-ingesting it.",
    )


class IngestedDocument(BaseModel):
    source: str
    chunks: int
    characters: int
    skipped_reason: str | None = None


class IngestResponse(BaseModel):
    documents: list[IngestedDocument]
    total_chunks: int
    collection: str


class ModelOption(BaseModel):
    """One selectable model.

    Shaped for a picker: `id` is what you send back as `model` on a
    request, everything else is display metadata. Annotations come from
    the curated catalogue and are None for models it doesn't cover.
    """

    id: str
    provider: str
    label: str | None = None
    context: str | None = None
    cost: str | None = None
    notes: str | None = None
    kind: str = Field(default="chat", description='"chat" or "embedding".')
    is_default: bool = Field(
        default=False, description="True if this is the provider's configured default."
    )


class ProviderModels(BaseModel):
    """Every model one provider can serve right now."""

    provider: str
    available: bool = Field(description="Reachable and usable at this moment.")
    is_default_provider: bool
    default_model: str | None
    requires_api_key: bool = Field(
        description="True for cloud providers; they're absent entirely when unconfigured."
    )
    models: list[ModelOption]


class ModelsResponse(BaseModel):
    """GET /api/models - what the model picker is built from.

    Model ids are queried live from each provider rather than hard-coded,
    so this never goes stale. Any `id` here is valid in the `model` field
    of a query request, paired with its `provider`.
    """

    default_provider: str
    providers: list[ProviderModels]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    providers: dict[str, bool]
    chat_model: str
    embed_model: str
    collection: str
    indexed_chunks: int
