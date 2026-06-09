# RAG AI Agent

A production-ready Retrieval-Augmented Generation (RAG) AI Agent with document ingestion, chat interface, and scalable architecture.

## Features

- **Document Ingestion**: Support for PDF, DOCX, TXT, CSV, HTML files
- **Automated PDF Processing**: Auto-extract text from PDFs
- **Database Integration**: SQLite with SQLAlchemy for metadata
- **Authentication**: JWT-based user authentication
- **Scalable Architecture**: Async processing, caching ready
- **Modern UI**: Glassmorphism design with responsive interface
- **API Endpoints**: RESTful API for all operations
- **Docker Deployment**: Containerized for easy deployment

## Quick Start

### Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```
4. Run the application:
   ```bash
   uvicorn main:app --reload
   ```
5. Open http://localhost:8000

### Docker Deployment

1. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

## API Documentation

- **Health Check**: `GET /health`
- **Authentication**: `POST /api/auth/login`, `POST /api/auth/register`
- **Documents**: `POST /api/documents/upload`, `GET /api/documents`
- **Chat**: `POST /api/chat/session`, `POST /api/chat/{session_id}/message`

## Security Features

- JWT authentication
- Input validation
- CORS protection
- File upload restrictions
- SQL injection prevention

## Scalability

- Async processing with FastAPI
- Vector database with ChromaDB
- Caching ready with Redis
- Background task processing with Celery

## Deployment to Production

1. Set environment variables in production
2. Use a production database (PostgreSQL)
3. Configure reverse proxy (nginx)
4. Set up monitoring and logging
5. Enable HTTPS

## Environment Variables

See `.env.example` for all required variables.