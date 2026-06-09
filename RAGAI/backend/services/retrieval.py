"""
Retrieval Service
Vector similarity search with MMR-based re-ranking for diversity.
"""
import logging
import math
from typing import List, Dict, Any, Tuple

from core.vectorstore import get_collection
from core.embeddings import embed_query
from core.config import settings

logger = logging.getLogger(__name__)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mmr_rerank(
    query_embedding: List[float],
    candidates: List[Dict[str, Any]],
    top_k: int,
    lambda_param: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance re-ranking.
    Balances relevance to the query vs. diversity among selected chunks.
    lambda_param: 1.0 = pure relevance, 0.0 = pure diversity
    """
    if not candidates:
        return []

    # Extract embeddings stored in candidate dicts
    selected: List[Dict[str, Any]] = []
    remaining = list(candidates)

    while remaining and len(selected) < top_k:
        best_score = -float("inf")
        best_idx = 0

        for i, cand in enumerate(remaining):
            relevance = cand.get("relevance_score", 0.0)
            if selected:
                max_sim = max(
                    _cosine_similarity(
                        cand.get("embedding", []),
                        sel.get("embedding", []),
                    )
                    for sel in selected
                )
                score = lambda_param * relevance - (1 - lambda_param) * max_sim
            else:
                score = relevance

            if score > best_score:
                best_score = score
                best_idx = i

        chosen = remaining.pop(best_idx)
        chosen["mmr_score"] = best_score
        selected.append(chosen)

    return selected


def retrieve_chunks(query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
    """
    Search ChromaDB for the most relevant chunks, then re-rank with MMR.
    Returns a list of dicts with keys: text, metadata, relevance_score, mmr_score.
    """
    fetch_k = (top_k or settings.top_k_results) * 3   # fetch extra for reranking pool
    final_k = top_k or settings.rerank_top_k

    try:
        q_emb = embed_query(query)
    except Exception as exc:
        logger.error(f"Embedding query failed: {exc}")
        raise

    collection = get_collection()
    total_docs = collection.count()
    if total_docs == 0:
        logger.warning("Vector store is empty — no documents ingested yet.")
        return []

    fetch_k = min(fetch_k, total_docs)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    candidates: List[Dict[str, Any]] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    embeddings = results.get("embeddings", [[]])[0]

    for doc, meta, dist, emb in zip(docs, metas, distances, embeddings):
        # ChromaDB cosine distance → similarity
        relevance = 1.0 - dist
        candidates.append({
            "text": doc,
            "metadata": meta,
            "relevance_score": relevance,
            "embedding": emb,
        })

    reranked = _mmr_rerank(q_emb, candidates, top_k=final_k)

    # Drop raw embedding from output (not needed downstream)
    for chunk in reranked:
        chunk.pop("embedding", None)

    logger.info(f"Retrieved {len(reranked)} chunks for query (pool={len(candidates)})")
    return reranked
