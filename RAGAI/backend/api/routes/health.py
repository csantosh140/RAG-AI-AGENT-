"""
Health Check Route
Returns component-level status for monitoring and readiness probes.
"""
import logging
from datetime import datetime, timezone
from fastapi import APIRouter

from api.schemas import HealthResponse, ComponentStatus
from core.config import settings
from services.memory import conversation_store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health_check():
    """Returns detailed health status of all system components."""
    components: list[ComponentStatus] = []
    overall = "healthy"
    doc_count = 0

    # 1. Gemini API key check
    if settings.gemini_api_key:
        components.append(ComponentStatus(name="Gemini API", status="ok", detail=f"Model: {settings.chat_model}"))
    else:
        components.append(ComponentStatus(name="Gemini API", status="error", detail="GEMINI_API_KEY not set"))
        overall = "degraded"

    # 2. ChromaDB check
    try:
        from core.vectorstore import get_collection
        collection = get_collection()
        doc_count = collection.count()
        components.append(ComponentStatus(
            name="ChromaDB",
            status="ok",
            detail=f"{doc_count} chunks stored in '{settings.chroma_collection_name}'"
        ))
    except Exception as exc:
        components.append(ComponentStatus(name="ChromaDB", status="error", detail=str(exc)))
        overall = "unhealthy"

    # 3. Memory store
    sessions = await conversation_store.list_sessions()
    components.append(ComponentStatus(
        name="Memory Store",
        status="ok",
        detail=f"{len(sessions)} active session(s)"
    ))

    return HealthResponse(
        status=overall,
        app_name=settings.app_name,
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        components=components,
        document_count=doc_count,
        active_sessions=len(sessions),
    )
