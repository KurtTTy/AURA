from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.analyst import DataSession, run_analyst
from app.deps import Depends, Settings, get_data_session, get_registry, get_settings
from app.llm_providers import Message, ProviderError, ProviderRegistry, UnknownProviderError
from app.models import Mode, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analyst", tags=["analyst"])

#: No data loaded. Mirrors rag.py's NO_CONTEXT_ANSWER - refusing plainly
#: beats inventing numbers from an empty database.
NO_DATA_ANSWER = (
    "No data is loaded, so I can't answer that. "
    "Load a file first - in the CLI: /load <path to .csv>"
)


async def analyze(
    request: QueryRequest,
    session: DataSession,
    registry: ProviderRegistry,
    settings: Settings,
) -> QueryResponse:
    """Run the analyst loop and shape the result for the API.

    Resolve provider and model BEFORE the empty-data check: both are
    required fields on QueryResponse, so an early return with None for
    either raises ValidationError instead of the friendly message.
    """
    provider = registry.get(request.provider)
    model_name = request.model or registry.default_model_for(provider.name) or ""

    if not session.table():
        return QueryResponse(
            answer=NO_DATA_ANSWER,
            mode=Mode.ANALYST,
            provider=provider.name,
            model=model_name,
        )

    # Boundary conversion: Pydantic in, dataclasses out. Keeps
    # app/analyst/ free of web-framework types.
    history = [Message(role=m.role, content=m.content) for m in request.history]

    result = await run_analyst(
        request.question,
        session,
        provider,
        model=model_name,
        history=history,
        max_turns=settings.analyst_max_turns,
    )

    if result.hit_limit:
        logger.warning(
            "Analyst hit the %d-turn limit on %r", settings.analyst_max_turns, request.question[:60]
        )

    return QueryResponse(
        answer=result.answer,
        mode=Mode.ANALYST,
        provider=provider.name,
        model=model_name,
        sql=result.sql,
        # str(): JSON has no path type.
        chart_path=str(result.chart_path) if result.chart_path else None,
        turns=result.turns,
        exports=[str(f) for f in result.exports],
    )


@router.post("", response_model=QueryResponse)
async def analyze_endpoint(
    request: QueryRequest,
    session: DataSession = Depends(get_data_session),
    registry: ProviderRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    """Ask a question about the loaded tabular data."""
    try:
        return await analyze(request, session, registry, settings)
    except UnknownProviderError as exc:
        # A bad provider name is the caller's mistake, not an outage -
        # retrying changes nothing, so 503 would be misleading advice.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderError as exc:
        # 503: our code is fine, the LLM backend isn't reachable.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
