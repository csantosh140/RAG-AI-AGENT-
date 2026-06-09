"""
RAG AI Agent — FastAPI Application Entry Point
"""
import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from core.config import settings

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm up ChromaDB connection."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    try:
        from core.vectorstore import get_collection
        col = get_collection()
        logger.info(f"ChromaDB ready — {col.count()} chunks in collection '{settings.chroma_collection_name}'")
    except Exception as exc:
        logger.warning(f"ChromaDB warm-up failed (will retry on first request): {exc}")

    if not settings.gemini_api_key:
        logger.warning("⚠  GEMINI_API_KEY is not set. Embedding endpoints will fail until configured.")
    else:
        logger.info("Gemini API key loaded ✓ (embeddings)")

    if not settings.anthropic_api_key:
        logger.warning("⚠  ANTHROPIC_API_KEY is not set. Chat endpoints will fail until configured.")
    else:
        logger.info("Anthropic Claude API key loaded ✓ (chat)")

    yield
    logger.info("Shutting down RAG AI Agent.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "A production-ready Retrieval-Augmented Generation (RAG) AI Agent. "
        "Upload documents, ask questions, and get grounded answers with source citations."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Frontend served from same host; wildcard for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from api.routes.health import router as health_router
from api.routes.documents import router as documents_router
from api.routes.chat import router as chat_router

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)

# ── Static Frontend ───────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")

# ── Dev Entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
