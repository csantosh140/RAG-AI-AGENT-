import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import uuid
from dotenv import load_dotenv

# Load .env from project root (parent of backend/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rag_agent.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(String, unique=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String)
    total_chunks = Column(Integer, default=0)

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, default=lambda: str(uuid.uuid4()))
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    role = Column(String) # "user" or "assistant"
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Context storing fields
    search_sources = Column(Text, nullable=True) # Stores JSON string of retrieved doc/web sources
    web_search_query = Column(Text, nullable=True) # Stores the query used for web search
    
    session = relationship("ChatSession", back_populates="messages")

Base.metadata.create_all(bind=engine)

# Run SQLite automated schema migrations safely
def run_migrations():
    from sqlalchemy import inspect, text
    try:
        inspector = inspect(engine)
        if 'chat_messages' not in inspector.get_table_names():
            return
            
        columns = [col['name'] for col in inspector.get_columns('chat_messages')]
        
        with engine.begin() as conn:
            if 'search_sources' not in columns:
                try:
                    conn.execute(text("ALTER TABLE chat_messages ADD COLUMN search_sources TEXT"))
                    print("Migration: Added search_sources column to chat_messages")
                except Exception as e:
                    print(f"Migration search_sources error: {e}")
            if 'web_search_query' not in columns:
                try:
                    conn.execute(text("ALTER TABLE chat_messages ADD COLUMN web_search_query TEXT"))
                    print("Migration: Added web_search_query column to chat_messages")
                except Exception as e:
                    print(f"Migration web_search_query error: {e}")
    except Exception as err:
        print(f"Migration inspect error (non-critical): {err}")

run_migrations()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def store_chat_message_with_search_data(
    db, 
    session_id: int, 
    role: str, 
    content: str, 
    search_sources: dict = None, 
    web_search_query: str = None
) -> ChatMessage:
    """
    Dedicated utility function to safely store chat history, messages, 
    and all associated search context/data in the SQL database.
    """
    import json
    sources_json = None
    if search_sources is not None:
        try:
            sources_json = json.dumps(search_sources)
        except Exception as e:
            print(f"Error serializing search sources: {e}")
            sources_json = str(search_sources)
            
    msg = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        search_sources=sources_json,
        web_search_query=web_search_query
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg
