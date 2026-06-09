"""
RAG Agent Service
Stateful multi-turn agent: query expansion → retrieval → grounded streaming generation.
Uses Google Gemini for chat generation.
"""
import asyncio
import logging
from typing import AsyncGenerator, List, Dict, Any

import google.generativeai as genai

from core.config import settings
from core.embeddings import _ensure_configured
from services.retrieval import retrieve_chunks
from services.memory import conversation_store

logger = logging.getLogger(__name__)

# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert RAG AI Assistant. Your job is to answer questions accurately and helpfully using ONLY the provided context documents.

CORE RULES:
1. Base your answer EXCLUSIVELY on the provided context. Do not use prior knowledge beyond what's given.
2. If the context does not contain enough information to answer, say so clearly — do NOT guess or fabricate.
3. Always cite your sources using [Source N] notation at the end of relevant sentences.
4. Keep answers clear, well-structured, and appropriately concise.
5. If the user's question is conversational or a greeting, respond naturally without forcing source citations.
6. When referencing specific data, quotes, or statistics, be precise.

FORMAT GUIDELINES:
- Use markdown formatting: bold for key terms, bullet lists for multiple points, code blocks for code.
- Structure complex answers with headers.
- End with a brief summary if the answer is long.

You maintain conversation context across turns for follow-up questions."""


# ─── Agent ────────────────────────────────────────────────────────────────────

def _build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block."""
    if not chunks:
        return "No relevant documents found in the knowledge base."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "Unknown")
        score = chunk.get("mmr_score", chunk.get("relevance_score", 0.0))
        parts.append(
            f"[Source {i}] — {filename} (relevance: {score:.2f})\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _build_history_text(history: list) -> str:
    """Serialise recent conversation into prompt-friendly format."""
    lines = []
    for msg in history[-6:]:   # last 3 turns
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _expand_query(query: str, history: list) -> str:
    """
    Simple rule-based query expansion using recent context.
    Prepends recent topic keywords if query is short/ambiguous.
    """
    if len(query.split()) >= 6:
        return query
    if len(history) >= 2:
        prev = history[-1].content if history else ""
        # If query is a follow-up (starts with pronoun/connector), enrich it
        follow_up_words = {"it", "this", "that", "they", "he", "she", "what", "why", "how", "explain more", "tell me more"}
        first_word = query.lower().split()[0] if query.split() else ""
        if first_word in follow_up_words and prev:
            # Append key nouns from previous answer (first 80 chars)
            query = f"{query} (context: {prev[:80]})"
    return query


def _build_gemini_prompt(history: list, context_block: str, user_query: str) -> str:
    """
    Build the prompt for Gemini.
    Includes system prompt, recent conversation history, and the current user turn with context.
    """
    parts = [f"System Instructions:\n{SYSTEM_PROMPT}"]
    
    # Add recent conversation history
    for msg in history[-6:]:
        role = "User" if msg.role == "user" else "Assistant"
        parts.append(f"{role}: {msg.content}")
    
    # Build the current user message with context
    user_content = (
        f"## Retrieved Context\n{context_block}\n\n"
        f"## Current Question\n{user_query}\n\n"
        f"## Your Answer"
    )
    parts.append(f"User: {user_content}")
    
    return "\n\n".join(parts)


async def stream_agent_response(
    session_id: str,
    user_query: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Main agent streaming generator.
    Yields dicts with keys:
      - {"type": "token", "content": "..."}  — partial text tokens
      - {"type": "sources", "sources": [...]} — final source citations
      - {"type": "error", "message": "..."}  — on failure
      - {"type": "done"}                      — stream end signal
    """
    # Ensure embeddings are configured
    _ensure_configured()

    # Validate Gemini API key
    if not settings.gemini_api_key:
        yield {"type": "error", "message": "GEMINI_API_KEY is not set. Please add it to backend/.env."}
        return

    # 1. Load history
    history = await conversation_store.get_history(session_id)

    # 2. Expand query for better retrieval
    expanded_query = _expand_query(user_query, history)
    logger.info(f"[{session_id}] Query: {user_query!r} → expanded: {expanded_query!r}")

    # 3. Retrieve relevant chunks
    try:
        chunks = await asyncio.get_event_loop().run_in_executor(
            None, retrieve_chunks, expanded_query, settings.top_k_results
        )
    except Exception as exc:
        logger.error(f"Retrieval failed: {exc}")
        yield {"type": "error", "message": f"Retrieval error: {exc}"}
        return

    # 4. Build grounded prompt
    context_block = _build_context_block(chunks)
    prompt = _build_gemini_prompt(history, context_block, user_query)

    # 5. Generate with Gemini
    genai.configure(api_key=settings.gemini_api_key)
    
    # Try available models ( Gemini model names change frequently )
    model_names = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-pro-latest"]
    assistant_reply = ""
    
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name=model_name)
            
            # Run generation in executor (synchronous API)
            def _generate():
                response = model.generate_content(prompt)
                return response.text
            
            assistant_reply = await asyncio.get_event_loop().run_in_executor(None, _generate)
            
            # Simulate streaming by yielding word by word
            words = assistant_reply.split()
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield {"type": "token", "content": chunk}
            
            break  # Success, exit loop
            
        except Exception as exc:
            logger.warning(f"Gemini model {model_name} failed: {exc}")
            if model_name == model_names[-1]:
                logger.error(f"All Gemini models failed: {exc}")
                yield {"type": "error", "message": f"Generation error: {exc}"}
                return
            continue  # Try next model

    # 6. Build source citations
    sources = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        sources.append({
            "index": i,
            "filename": meta.get("filename", "Unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "total_chunks": meta.get("total_chunks", 0),
            "doc_id": meta.get("doc_id", ""),
            "relevance_score": round(chunk.get("relevance_score", 0.0), 4),
            "snippet": chunk["text"][:200] + ("..." if len(chunk["text"]) > 200 else ""),
        })

    # 7. Persist to memory
    await conversation_store.add_message(session_id, "user", user_query)
    await conversation_store.add_message(session_id, "assistant", assistant_reply, sources=sources)

    yield {"type": "sources", "sources": sources}
    yield {"type": "done"}
    logger.info(f"[{session_id}] Response complete ({len(assistant_reply)} chars, {len(sources)} sources)")

