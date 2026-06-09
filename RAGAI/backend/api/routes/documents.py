"""
Document API Routes
Upload, list, and delete ingested documents.
"""
import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from services.ingestion import ingest_document, list_documents, delete_document, reset_collection_data
from api.schemas import UploadResponse, DocumentInfo, DeleteResponse, ResetResponse
from core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["Documents"])

MAX_BYTES = settings.max_file_size_mb * 1024 * 1024
ALLOWED_EXT = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".html", ".htm"}


@router.post("/upload", response_model=UploadResponse, summary="Upload one or more documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more documents for ingestion into the knowledge base.
    Supported formats: PDF, DOCX, DOC, TXT, MD, CSV, HTML.
    """
    results: List[DocumentInfo] = []
    errors: List[str] = []

    for file in files:
        try:
            content = await file.read()

            if len(content) > MAX_BYTES:
                errors.append(f"{file.filename}: exceeds {settings.max_file_size_mb} MB limit")
                continue

            if not content:
                errors.append(f"{file.filename}: file is empty")
                continue

            info = ingest_document(file.filename or "upload", content)
            results.append(DocumentInfo(
                doc_id=info["doc_id"],
                filename=info["filename"],
                total_chunks=info["chunks"],
            ))
            logger.info(f"Uploaded: {file.filename} → {info['chunks']} chunks")

        except ValueError as ve:
            errors.append(f"{file.filename}: {ve}")
        except Exception as exc:
            logger.exception(f"Ingestion failed for {file.filename}")
            errors.append(f"{file.filename}: internal error — {exc}")

    return UploadResponse(
        success=len(results) > 0,
        documents=results,
        errors=errors,
    )


@router.get("/", response_model=List[DocumentInfo], summary="List all ingested documents")
async def get_documents():
    """Return a list of all documents currently in the knowledge base."""
    try:
        docs = list_documents()
        return [DocumentInfo(**d) for d in docs]
    except Exception as exc:
        logger.exception("Failed to list documents")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/reset", response_model=ResetResponse, summary="Wipe entire knowledge base")
async def reset_knowledge_base():
    """Delete ALL documents and reset the vector store. This action is irreversible."""
    try:
        reset_collection_data()
        return ResetResponse(success=True, message="Knowledge base has been reset successfully.")
    except Exception as exc:
        logger.exception("Reset failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/{doc_id}", response_model=DeleteResponse, summary="Delete a specific document")
async def delete_single_document(doc_id: str):
    """Delete all chunks of a specific document from the knowledge base."""
    try:
        deleted = delete_document(doc_id)
        if deleted == 0:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
        return DeleteResponse(success=True, doc_id=doc_id, chunks_deleted=deleted)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Delete failed for doc_id={doc_id}")
        raise HTTPException(status_code=500, detail=str(exc))
