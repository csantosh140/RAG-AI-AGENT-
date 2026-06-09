import chromadb
from chromadb.config import Settings as ChromaSettings
from core.config import settings
import logging

logger = logging.getLogger(__name__)

_client = None
_collection = None


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB client initialised at '{settings.chroma_persist_dir}'")
    return _client


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Using ChromaDB collection '{settings.chroma_collection_name}'")
    return _collection


def reset_collection() -> None:
    """Drop and recreate the collection (destructive)."""
    global _collection
    client = get_client()
    try:
        client.delete_collection(settings.chroma_collection_name)
    except Exception:
        pass
    _collection = None
    get_collection()
    logger.info("Collection reset.")
