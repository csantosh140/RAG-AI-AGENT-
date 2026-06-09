import os
from dotenv import load_dotenv

# Load .env from project root (parent of backend/services/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
# pyrefly: ignore [missing-import]
import PyPDF2
import docx
import csv
from io import BytesIO

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

def extract_pages_from_file(file_content: bytes, filename: str) -> List[Dict[str, Any]]:
    ext = filename.split(".")[-1].lower()
    pages = []
    try:
        if ext == "pdf":
            reader = PyPDF2.PdfReader(BytesIO(file_content))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages.append({"text": text, "page": i + 1})
        elif ext in ["docx", "doc"]:
            doc = docx.Document(BytesIO(file_content))
            text = "\n".join([para.text for para in doc.paragraphs])
            pages.append({"text": text, "page": 1})
        elif ext == "csv":
            decoded = file_content.decode('utf-8').splitlines()
            reader = csv.reader(decoded)
            text = "\n".join([" ".join(row) for row in reader])
            pages.append({"text": text, "page": 1})
        else: # txt, md, html etc
            text = file_content.decode('utf-8')
            pages.append({"text": text, "page": 1})
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
    return pages

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def process_and_store_document(doc_id: str, filename: str, file_content: bytes):
    pages = extract_pages_from_file(file_content, filename)
    if not pages:
        return 0
    
    all_chunks = []
    all_metadatas = []
    
    for p in pages:
        page_chunks = chunk_text(p["text"])
        for chunk in page_chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"page": p["page"]})
            
    if not all_chunks:
        return 0

    ids = [f"{doc_id}_{i}" for i in range(len(all_chunks))]
    metadatas = [{"doc_id": doc_id, "filename": filename, "index": i, "page": m["page"]} for i, m in enumerate(all_metadatas)]
    
    collection.add(
        documents=all_chunks,
        metadatas=metadatas,
        ids=ids
    )
    return len(all_chunks)

def delete_document_chunks(doc_id: str):
    # ChromaDB supports deleting by metadata in newer versions, 
    # but we can also query and delete or just use where clause
    collection.delete(where={"doc_id": doc_id})

def reset_all_documents():
    # Simplest way is to just delete everything or delete collection and recreate
    global collection
    chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

from typing import List, Dict, Any, Optional

def search_documents(query: str, top_k: int = int(os.getenv("TOP_K_RESULTS", 5)), filenames: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    where_clause = None
    if filenames:
        if len(filenames) == 1:
            where_clause = {"filename": filenames[0]}
        else:
            where_clause = {"filename": {"$in": filenames}}

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_clause
    )
    
    sources = []
    if results['documents'] and len(results['documents'][0]) > 0:
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        distances = results['distances'][0] if 'distances' in results and results['distances'] else [0.0]*len(docs)
        
        for i in range(len(docs)):
            sources.append({
                "index": i + 1,
                "filename": metas[i].get('filename', 'Unknown'),
                "page": metas[i].get('page', 1),
                "snippet": docs[i][:200] + "..." if len(docs[i]) > 200 else docs[i],
                "content": docs[i],
                "relevance_score": max(0.0, 1.0 - distances[i]) # Simple conversion
            })
    return sources

def search_topics(keyword: str, top_k: int = 20) -> Dict[str, Any]:
    """Search all document chunks for a keyword/topic and return grouped results."""
    # 1. Semantic search via ChromaDB
    results = collection.query(
        query_texts=[keyword],
        n_results=top_k
    )
    
    topics = []
    seen_snippets = set()
    
    if results['documents'] and len(results['documents'][0]) > 0:
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        distances = results['distances'][0] if 'distances' in results and results['distances'] else [0.0] * len(docs)
        
        for i in range(len(docs)):
            text = docs[i]
            relevance = max(0.0, 1.0 - distances[i])
            
            # Skip very low relevance results
            if relevance < 0.15:
                continue
            
            # Deduplicate near-identical snippets
            snippet_key = text[:100].strip().lower()
            if snippet_key in seen_snippets:
                continue
            seen_snippets.add(snippet_key)
            
            # Find context around the keyword (case-insensitive)
            keyword_lower = keyword.lower()
            text_lower = text.lower()
            keyword_pos = text_lower.find(keyword_lower)
            
            if keyword_pos >= 0:
                # Extract surrounding context (200 chars around keyword)
                start = max(0, keyword_pos - 100)
                end = min(len(text), keyword_pos + len(keyword) + 100)
                highlight_snippet = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
                has_exact_match = True
            else:
                # No exact match but semantically relevant
                highlight_snippet = text[:250] + ("..." if len(text) > 250 else "")
                has_exact_match = False
            
            topics.append({
                "index": len(topics) + 1,
                "filename": metas[i].get('filename', 'Unknown'),
                "doc_id": metas[i].get('doc_id', ''),
                "page": metas[i].get('page', 1),
                "snippet": highlight_snippet,
                "full_text": text,
                "relevance_score": round(relevance, 3),
                "has_exact_match": has_exact_match,
            })
    
    # Sort: exact matches first, then by relevance
    topics.sort(key=lambda x: (-int(x['has_exact_match']), -x['relevance_score']))
    
    # Re-index after sorting
    for i, t in enumerate(topics):
        t['index'] = i + 1
    
    # Group by document for the summary
    doc_groups = {}
    for t in topics:
        fname = t['filename']
        if fname not in doc_groups:
            doc_groups[fname] = {"filename": fname, "doc_id": t['doc_id'], "sections": [], "pages": set()}
        doc_groups[fname]["sections"].append(t)
        doc_groups[fname]["pages"].add(t['page'])
    
    # Convert sets to sorted lists for JSON serialization
    grouped = []
    for fname, group in doc_groups.items():
        grouped.append({
            "filename": fname,
            "doc_id": group["doc_id"],
            "pages": sorted(group["pages"]),
            "section_count": len(group["sections"]),
            "sections": group["sections"]
        })
    
    return {
        "keyword": keyword,
        "total_results": len(topics),
        "results": topics,
        "grouped_by_document": grouped,
    }


def get_document_text(doc_id: str = None) -> str:
    """Retrieve text from chunks, up to a limit, for full-document tasks like quiz generation."""
    try:
        where_clause = {"doc_id": doc_id} if doc_id else None
        results = collection.get(where=where_clause)
        if not results or not results['documents']:
            return ""
        
        # Combine chunks, limit to first 30 chunks to avoid exceeding context window
        text = "\n\n".join(results['documents'][:30])
        return text
    except Exception as e:
        print(f"Error retrieving text: {e}")
        return ""

