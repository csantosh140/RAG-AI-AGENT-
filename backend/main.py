from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import json
import uuid

from dotenv import load_dotenv
# Load .env from project root (parent of backend/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Import as absolute modules so the app can run both as `backend.main:app` and `main:app`.
# This avoids `ImportError: attempted relative import with no known parent package`.
from database import engine, get_db, Document, ChatSession, ChatMessage
from schemas import ChatMessageCreate, DocumentResponse
from services.retrieval import process_and_store_document, delete_document_chunks, reset_all_documents, search_topics
from services.agent import generate_chat_response, generate_quiz_response
from services.web_search import fetch_web_context
from services.image_gen import get_image_path, IMAGE_DIR



app = FastAPI(title="RAG AI Agent API")

CORS_ORIGINS = json.loads(os.getenv("CORS_ORIGINS", '["*"]'))
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    doc_count = db.query(Document).count()
    return {"status": "healthy", "document_count": doc_count}

@app.get("/debug-key")
def debug_key():
    return {
        "exists": bool(os.getenv("GEMINI_API_KEY"))
    }

@app.get("/api/documents/", response_model=List[DocumentResponse])
def get_documents(db: Session = Depends(get_db)):
    return db.query(Document).all()

@app.post("/api/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    uploaded_docs = []
    errors = []
    
    for file in files:
        try:
            content = await file.read()
            doc_id = str(uuid.uuid4())
            
            # Process and store in ChromaDB
            chunks_count = process_and_store_document(doc_id, file.filename, content)
            
            if chunks_count > 0:
                # Store metadata in SQLite
                new_doc = Document(doc_id=doc_id, filename=file.filename, total_chunks=chunks_count)
                db.add(new_doc)
                uploaded_docs.append(file.filename)
            else:
                errors.append(f"Could not extract text from {file.filename}")
                
        except Exception as e:
            errors.append(f"Error processing {file.filename}: {str(e)}")
            
    db.commit()
    return {"documents": uploaded_docs, "errors": errors}

@app.delete("/api/documents/reset")
def reset_documents(db: Session = Depends(get_db)):
    # Delete from sqlite
    db.query(Document).delete()
    db.commit()
    # Delete from chroma (simplified, ideally filter by user)
    reset_all_documents()
    return {"status": "success"}

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete from chroma
    delete_document_chunks(doc_id)
    
    # Delete from sqlite
    db.delete(doc)
    db.commit()
    return {"status": "success"}

@app.post("/api/chat/session")
def create_chat_session(db: Session = Depends(get_db)):
    session = ChatSession()
    db.add(session)
    db.commit()
    db.refresh(session)
    return {"session_id": session.session_id}

@app.get("/api/chat/history")
def get_all_chat_sessions(db: Session = Depends(get_db)):
    """Fetch all chat sessions from SQLite database with summary info."""
    sessions = db.query(ChatSession).all()
    # Sort sessions by the latest message ID (most recent activity first)
    sessions = sorted(
        sessions,
        key=lambda s: max((m.id for m in s.messages), default=-1),
        reverse=True
    )
    result = []
    for s in sessions:
        msgs = []
        first_user_msg = None
        for m in sorted(s.messages, key=lambda x: x.id):
            msgs.append({
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None
            })
            if m.role == "user" and not first_user_msg:
                first_user_msg = m.content
                
        # Truncate first user message for a clean title
        if first_user_msg:
            title = first_user_msg[:45] + "..." if len(first_user_msg) > 45 else first_user_msg
        else:
            title = "New Conversation"
            
        result.append({
            "session_id": s.session_id,
            "title": title,
            "messages_count": len(msgs),
            "last_message": msgs[-1]["content"] if msgs else None
        })
    return result


@app.get("/api/chat/session/{session_id}")
def get_chat_session_details(session_id: str, db: Session = Depends(get_db)):
    """Fetch details of a specific chat session including messages and full search history/context."""
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = []
    for m in sorted(session.messages, key=lambda x: x.id):
        sources = None
        if m.search_sources:
            try:
                sources = json.loads(m.search_sources)
            except Exception:
                sources = m.search_sources
                
        messages.append({
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "search_sources": sources,
            "web_search_query": m.web_search_query
        })
        
    return {
        "session_id": session.session_id,
        "messages": messages
    }

@app.post("/api/chat/{session_id}/message")
def send_chat_message(
    session_id: str,
    message: ChatMessageCreate,
    web_search: bool = Query(default=True, description="Enable web search augmentation"),
    db: Session = Depends(get_db)
):
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Save user message (strip attached files tag for a clean database record)
    import re
    clean_message = re.sub(r'^\[AttachedFiles:\s*.*?\s*\]\s*', '', message.message).strip()
    user_msg = ChatMessage(session_id=session.id, role="user", content=clean_message)
    db.add(user_msg)
    db.commit()
    
    # Get history
    history = db.query(ChatMessage).filter(ChatMessage.session_id == session.id).order_by(ChatMessage.created_at).all()
    
    # Generate streaming response with optional web search augmentation
    return StreamingResponse(
        generate_chat_response(message.message, session.id, db, history, web_search=web_search),
        media_type="text/event-stream"
    )

@app.delete("/api/chat/{session_id}")
def delete_chat_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db.delete(session)
    db.commit()
    return {"status": "success"}

@app.get("/api/documents/search-topics")
def search_topics_endpoint(keyword: str):
    if not keyword or not keyword.strip():
        raise HTTPException(status_code=400, detail="Keyword is required")
    results = search_topics(keyword.strip())
    return results

@app.get("/api/web-search")
def web_search_endpoint(query: str, max_results: int = Query(default=3, le=10)):
    """Standalone web search endpoint for testing or direct use."""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    result = fetch_web_context(query.strip(), max_results=max_results)
    return result

@app.get("/api/quiz/generate")
def generate_quiz(doc_id: str = None):
    return StreamingResponse(
        generate_quiz_response(doc_id),
        media_type="text/event-stream"
    )

@app.post("/api/documents/generate-pdf")
async def generate_pdf_endpoint(
    topic: str = Query(..., description="The topic to generate a PDF report for"),
    db: Session = Depends(get_db)
):
    from services.pdf_generator import generate_pdf_on_topic
    from services.retrieval import process_and_store_document
    
    if not topic or not topic.strip():
        raise HTTPException(status_code=400, detail="Topic is required")
        
    try:
        pdf_bytes, filename = generate_pdf_on_topic(topic.strip())
        
        doc_id = str(uuid.uuid4())
        chunks_count = process_and_store_document(doc_id, filename, pdf_bytes)
        
        if chunks_count > 0:
            new_doc = Document(doc_id=doc_id, filename=filename, total_chunks=chunks_count)
            db.add(new_doc)
            db.commit()
            db.refresh(new_doc)
        else:
            raise ValueError("Failed to index the generated PDF document.")
            
        from fastapi.responses import Response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Access-Control-Expose-Headers": "X-Document-ID, X-Document-Filename, X-Chunks-Count",
                "X-Document-ID": doc_id,
                "X-Document-Filename": filename,
                "X-Chunks-Count": str(chunks_count)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Generation failed: {str(e)}")

@app.get("/api/images/{image_id}")
def get_generated_image(image_id: str):
    """Serve a generated chart/graph image."""
    filepath = get_image_path(image_id)
    if not filepath:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath, media_type="image/png", filename=f"{image_id}.png")

# Mount generated images directory
os.makedirs(IMAGE_DIR, exist_ok=True)
app.mount("/generated_images", StaticFiles(directory=IMAGE_DIR), name="generated_images")

# Mount frontend
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "../frontend"), html=True), name="frontend")
