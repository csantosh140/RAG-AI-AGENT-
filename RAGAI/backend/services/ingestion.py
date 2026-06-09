"""
Document Ingestion Service
Parses uploaded files into text chunks and stores them in ChromaDB.
"""
import uuid
import logging
import io
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import settings
from core.vectorstore import get_collection
from core.embeddings import embed_texts

logger = logging.getLogger(__name__)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ── Parsers ──────────────────────────────────────────────────────────────────

def _parse_pdf(content: bytes) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(stream=content, filetype="pdf")
    return "\n\n".join(page.get_text() for page in doc)


def _parse_docx(content: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(content))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _parse_txt(content: bytes) -> str:
    return content.decode("utf-8", errors="replace")


def _parse_csv(content: bytes) -> str:
    import pandas as pd
    df = pd.read_csv(io.BytesIO(content))
    return df.to_string(index=False)


def _parse_html(content: bytes) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(content, "lxml")
    return soup.get_text(separator="\n")


PARSERS = {
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".doc": _parse_docx,
    ".txt": _parse_txt,
    ".md": _parse_txt,
    ".csv": _parse_csv,
    ".html": _parse_html,
    ".htm": _parse_html,
}


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_document(filename: str, content: bytes) -> Dict[str, Any]:
    """Parse, chunk, embed and store a document. Returns ingestion summary."""
    ext = Path(filename).suffix.lower()
    parser = PARSERS.get(ext)
    if not parser:
        raise ValueError(f"Unsupported file type: '{ext}'. Supported: {list(PARSERS.keys())}")

    logger.info(f"Parsing '{filename}' ({len(content)} bytes)")
    raw_text = parser(content)

    if not raw_text.strip():
        raise ValueError("Document appears to be empty or could not be parsed.")

    chunks = text_splitter.split_text(raw_text)
    logger.info(f"Split into {len(chunks)} chunks")

    doc_id = str(uuid.uuid4())
    ids, embeddings, documents, metadatas = [], [], [], []

    # Embed in batches of 10 to stay within rate limits
    batch_size = 10
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start: batch_start + batch_size]
        batch_embeddings = embed_texts(batch)
        for i, (chunk, emb) in enumerate(zip(batch, batch_embeddings)):
            chunk_id = f"{doc_id}_{batch_start + i}"
            ids.append(chunk_id)
            embeddings.append(emb)
            documents.append(chunk)
            metadatas.append({
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": batch_start + i,
                "total_chunks": len(chunks),
            })

    collection = get_collection()
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    logger.info(f"Ingested '{filename}' → doc_id={doc_id}, chunks={len(chunks)}")
    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks": len(chunks),
        "characters": len(raw_text),
    }


def list_documents() -> List[Dict[str, Any]]:
    """Return a deduplicated list of ingested documents."""
    collection = get_collection()
    result = collection.get(include=["metadatas"])
    seen, docs = set(), []
    for meta in result["metadatas"] or []:
        doc_id = meta.get("doc_id")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            docs.append({
                "doc_id": doc_id,
                "filename": meta.get("filename", "unknown"),
                "total_chunks": meta.get("total_chunks", 0),
            })
    return docs


def delete_document(doc_id: str) -> int:
    """Delete all chunks belonging to a document. Returns number of chunks deleted."""
    collection = get_collection()
    result = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
    ids_to_delete = result["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def reset_collection_data() -> None:
    """Drop and recreate the collection — wipes ALL documents (destructive)."""
    from core.vectorstore import reset_collection
    reset_collection()
    logger.info("Knowledge base has been fully reset.")
