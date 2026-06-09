@echo off
echo Starting RAG AI Agent Server...
echo.
echo Server will be available at: http://127.0.0.1:8000
echo Press CTRL+C to stop the server.
echo.
cd backend
..\.venv\Scripts\uvicorn main:app --port 8000 --reload
