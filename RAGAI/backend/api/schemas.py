"""
API Pydantic Schemas
Request/response models for all API endpoints.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    total_chunks: int


class UploadResponse(BaseModel):
    success: bool
    documents: List[DocumentInfo]
    errors: List[str] = []


class DeleteResponse(BaseModel):
    success: bool
    doc_id: str
    chunks_deleted: int


class ResetResponse(BaseModel):
    success: bool
    message: str


# ── Chat ──────────────────────────────────────────────────────────────────────

class SessionCreateResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")


class SourceCitation(BaseModel):
    index: int
    filename: str
    chunk_index: int
    total_chunks: int
    doc_id: str
    relevance_score: float
    snippet: str


class ChatMessageOut(BaseModel):
    role: str
    content: str
    timestamp: str
    sources: List[SourceCitation] = []


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessageOut]
    total_messages: int


# ── Health ────────────────────────────────────────────────────────────────────

class ComponentStatus(BaseModel):
    name: str
    status: str        # "ok" | "error" | "warning"
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str        # "healthy" | "degraded" | "unhealthy"
    app_name: str
    version: str
    timestamp: str
    components: List[ComponentStatus]
    document_count: int
    active_sessions: int
