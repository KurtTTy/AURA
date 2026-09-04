from __future__ import annotations

from fastapi import Depends, Request

from app.analyst import DataSession
from app.config import Settings, get_settings
from app.llm_providers import ProviderRegistry
from app.rag import VectorStore


def get_registry(request: Request) -> ProviderRegistry:
    """The process-wide LLM provider registry."""
    return request.app.state.registry


def get_store(request: Request) -> VectorStore:
    """The process-wide Chroma vector store."""
    return request.app.state.store


def get_data_session(request: Request) -> DataSession:
    """The process-wide DuckDB session. Per-request would start empty
    each time, losing every loaded table."""
    return request.app.state.data_session


# Re-exported so routers import every dependency from one place.
__all__ = [
    "Depends",
    "Settings",
    "get_data_session",
    "get_registry",
    "get_settings",
    "get_store",
]
