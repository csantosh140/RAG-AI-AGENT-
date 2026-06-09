"""
Chat API Routes
Session management and SSE streaming chat.
"""
import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from services.agent import stream_agent_response
from services.memory import conversation_store
from api.schemas import (
    SessionCreateResponse,
    ChatRequest,
    HistoryResponse,
    ChatMessageOut,
    SourceCitation,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("/session", response_model=SessionCreateResponse, summary="Create a new chat session")
async def create_session():
    """Create a new conversation session. Returns a session_id to use in subsequent requests."""
    session_id = await conversation_store.create_session()
    return SessionCreateResponse(session_id=session_id)


@router.post("/{session_id}/message", summary="Send a message (SSE streaming response)")
async def chat_message(session_id: str, request: ChatRequest):
    """
    Send a user message and receive a streaming SSE response.

    SSE event types:
    - `token`  — partial text token: `{"type": "token", "content": "..."}`
    - `sources` — source citations: `{"type": "sources", "sources": [...]}`
    - `error`  — error occurred: `{"type": "error", "message": "..."}`
    - `done`   — stream complete: `{"type": "done"}`
    """
    if not await conversation_store.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Create one first.")

    async def event_generator():
        try:
            async for event in stream_agent_response(session_id, request.message):
                yield {
                    "data": json.dumps(event),
                    "event": event.get("type", "message"),
                }
        except Exception as exc:
            logger.exception(f"Stream error in session {session_id}")
            yield {
                "data": json.dumps({"type": "error", "message": str(exc)}),
                "event": "error",
            }

    return EventSourceResponse(event_generator())


@router.get("/{session_id}/history", response_model=HistoryResponse, summary="Get conversation history")
async def get_history(session_id: str):
    """Return the full conversation history for a session."""
    if not await conversation_store.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    messages = await conversation_store.get_history(session_id)
    out = []
    for m in messages:
        sources = [SourceCitation(**s) for s in m.sources] if m.sources else []
        out.append(ChatMessageOut(
            role=m.role,
            content=m.content,
            timestamp=m.timestamp,
            sources=sources,
        ))
    return HistoryResponse(
        session_id=session_id,
        messages=out,
        total_messages=len(out),
    )


@router.delete("/{session_id}", summary="Delete a chat session")
async def delete_session(session_id: str):
    """Clear a conversation session and its history."""
    deleted = await conversation_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"success": True, "session_id": session_id, "message": "Session deleted."}
