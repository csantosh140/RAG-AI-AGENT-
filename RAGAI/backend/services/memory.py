"""
Conversation Memory Service
Thread-safe in-memory store for multi-turn chat history.
"""
import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str          # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    sources: List[Dict] = field(default_factory=list)


class ConversationStore:
    """Thread-safe in-memory conversation history store."""

    def __init__(self) -> None:
        self._store: Dict[str, List[Message]] = {}
        self._lock = asyncio.Lock()

    async def create_session(self) -> str:
        import uuid
        session_id = str(uuid.uuid4())
        async with self._lock:
            self._store[session_id] = []
        logger.info(f"Session created: {session_id}")
        return session_id

    async def session_exists(self, session_id: str) -> bool:
        async with self._lock:
            return session_id in self._store

    async def add_message(self, session_id: str, role: str, content: str, sources: Optional[List[Dict]] = None) -> None:
        async with self._lock:
            if session_id not in self._store:
                self._store[session_id] = []
            self._store[session_id].append(
                Message(role=role, content=content, sources=sources or [])
            )
            # Trim to max history (pairs of user+assistant)
            max_msgs = settings.max_chat_history * 2
            if len(self._store[session_id]) > max_msgs:
                self._store[session_id] = self._store[session_id][-max_msgs:]

    async def get_history(self, session_id: str) -> List[Message]:
        async with self._lock:
            return list(self._store.get(session_id, []))

    async def delete_session(self, session_id: str) -> bool:
        async with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                logger.info(f"Session deleted: {session_id}")
                return True
            return False

    async def list_sessions(self) -> List[str]:
        async with self._lock:
            return list(self._store.keys())


# Singleton
conversation_store = ConversationStore()
