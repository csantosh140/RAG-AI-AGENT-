from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    # App
    app_name: str = Field(default="RAG AI Agent", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")

    # Google Gemini (embeddings)
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")

    # Anthropic Claude (chat/generation)
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")

    # ChromaDB
    chroma_persist_dir: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(default="rag_documents", env="CHROMA_COLLECTION_NAME")

    # RAG
    chunk_size: int = Field(default=800, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")
    rerank_top_k: int = Field(default=3, env="RERANK_TOP_K")
    embedding_model: str = Field(default="models/text-embedding-004", env="EMBEDDING_MODEL")
    chat_model: str = Field(default="claude-sonnet-4-20250514", env="CHAT_MODEL")
    
    # Reranking
    use_reranking: bool = Field(default=False, env="USE_RERANKING")
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", env="RERANK_MODEL")

    # Memory
    memory_window_size: int = Field(default=10, env="MEMORY_WINDOW_SIZE")
    max_chat_history: int = Field(default=20, env="MAX_CHAT_HISTORY")

    # API
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:5500", "http://localhost:8000", "null", "file://"],
        env="CORS_ORIGINS",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
