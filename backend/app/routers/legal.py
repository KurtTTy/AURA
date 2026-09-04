from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models import QueryRequest, QueryResponse

router = APIRouter(prefix="/api/legal", tags=["legal"])

DISCLAIMER = (
    "This response comes from a small fine-tuned model and is informational "
    "only. It is not legal advice and should not be relied on as such."
)


@router.post("", response_model=QueryResponse)
async def legal_query(request: QueryRequest) -> QueryResponse:
    """Not implemented."""
    raise HTTPException(
        status_code=501,
        detail="The legal model is not available in this build.",
    )
