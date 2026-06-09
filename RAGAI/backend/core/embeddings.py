from sentence_transformers import SentenceTransformer
from core.config import settings
import logging
from typing import List

logger = logging.getLogger(__name__)

_model = None


def _ensure_configured():
    """No-op for local sentence-transformers models (no API key needed)."""
    pass


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # Map any Google model name to a valid local sentence-transformers model
        model_name = settings.embedding_model
        if model_name.startswith("models/") or not model_name:
            model_name = "all-MiniLM-L6-v2"
        logger.info(f"Loading local embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a list of texts using a local sentence-transformers model."""
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """Return a single query embedding optimised for retrieval."""
    model = _get_model()
    embedding = model.encode([query], convert_to_numpy=True)
    return embedding[0].tolist()


def get_chat_model():
    """Kept for backward compatibility; chat uses Anthropic Claude instead."""
    raise NotImplementedError("Chat generation is handled by Anthropic Claude, not Gemini.")

