import os
import json
import time
import google.generativeai as genai
from typing import List, Dict, Any, AsyncGenerator, Optional
from sqlalchemy.orm import Session
from database import ChatMessage
from services.retrieval import search_documents
from services.image_gen import generate_ai_photo

# Import the enhanced LLM engine
from services.llm_engine import generate_enhanced_response

def generate_image(prompt: str, width: int = 1024, height: int = 1024) -> Optional[str]:
    """
    Generate an AI photo/image from a text prompt using Pollinations.ai.
    Downloads the image and returns the local image_id.
    """
    return generate_ai_photo(prompt, width, height)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Broad fallback list — each model has its own separate free-tier quota
# Ordered by preference: newest / most capable first
PREFERRED_MODELS = [
    'models/gemini-2.5-flash-lite',
    'models/gemini-2.5-flash',
    'models/gemini-2.0-flash',
    'models/gemini-2.0-flash-lite',
    'models/gemini-1.5-flash-latest',
    'models/gemini-1.5-flash',
    'models/gemini-1.5-pro-latest',
    'models/gemini-1.0-pro',
    'models/gemini-pro',
]

MAX_RETRIES = len(PREFERRED_MODELS) + 1
RETRY_DELAY = 5  # seconds between quota-exhausted retries

def _get_available_models():
    """Return list of model names that support content generation."""
    return [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]

def _pick_model(available_models, skip_models=None):
    """Pick the best model from the preferred list, optionally skipping some."""
    skip = skip_models or set()
    for m in PREFERRED_MODELS:
        if m in available_models and m not in skip:
            return m
    # Fallback: pick any available model not in skip list
    for m in available_models:
        if m not in skip:
            return m
    return None

def _is_quota_error(error):
    """Check if an exception is a 429 quota/rate-limit error."""
    err_str = str(error).lower()
    return '429' in err_str or 'quota' in err_str or 'rate' in err_str or 'resource exhausted' in err_str


async def generate_chat_response(
    query: str, 
    session_id: int, 
    db: Session,
    history: List[ChatMessage],
    web_search: bool = True
) -> AsyncGenerator[str, None]:
    """
    Enhanced chat response generation using the LLM Engine.
    Combines document retrieval + web search + Gemini LLM for
    accurate, comprehensive, and up-to-date answers.
    
    Set web_search=False to disable web augmentation.
    """
    async for event in generate_enhanced_response(
        query=query,
        session_id=session_id,
        db=db,
        history=history,
        web_search_enabled=web_search
    ):
        yield event

async def generate_quiz_response(doc_id: str = None) -> AsyncGenerator[str, None]:
    from services.retrieval import get_document_text
    context = get_document_text(doc_id)
    
    if not context:
        yield f"data: {json.dumps({'type': 'error', 'message': 'No document text found. Please upload documents first.'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return
        
    prompt = (
        "You are an expert educator. Based on the following document context, "
        "generate a comprehensive quiz.\n\n"
        "The quiz must include:\n"
        "1. 3 Multiple Choice Questions (with options A, B, C, D)\n"
        "2. 3 True/False Questions\n"
        "3. 2 Short Answer Questions\n"
        "4. 2 Interview Questions\n\n"
        "Format the output cleanly in Markdown, providing the questions first, "
        "and then an 'Answer Key' section at the end.\n\n"
        f"CONTEXT:\n{context}\n"
    )
    
    if not GEMINI_API_KEY:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Gemini API key is missing. Please set it in .env'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    try:
        available_models = _get_available_models()
        skip_models = set()
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            selected_model = _pick_model(available_models, skip_models)
            
            if not selected_model:
                raise ValueError("__quota_exhausted__")
            
            try:
                if attempt > 0:
                    model_name = selected_model.split("/")[-1]
                    # Show as a status indicator, NOT as chat text
                    yield f"data: {json.dumps({'type': 'status', 'message': f'⏳ Switching to {model_name}...'})}\n\n"
                
                model = genai.GenerativeModel(selected_model)
                response = model.generate_content(prompt, stream=True)
                
                for chunk in response:
                    if chunk.text:
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.text})}\n\n"
                        
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return  # Success
                
            except Exception as model_err:
                last_error = model_err
                if _is_quota_error(model_err):
                    skip_models.add(selected_model)
                    model_name = selected_model.split('/')[-1]
                    print(f"[Quiz Retry {attempt+1}/{MAX_RETRIES}] Quota hit on {model_name}, trying next model...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise model_err
        
        raise last_error or ValueError("__quota_exhausted__")
        
    except Exception as e:
        err_msg = str(e)
        if '__quota_exhausted__' in err_msg or _is_quota_error(e):
            friendly = (
                "⚠️ **All Gemini models are temporarily at capacity** (free-tier quota reached).\n\n"
                "Please try one of the following:\n"
                "- Wait a minute and try again\n"
                "- Add a paid Gemini API key to your `.env` file\n"
                "- Visit [Google AI Studio](https://aistudio.google.com) to check your quota"
            )
            yield f"data: {json.dumps({'type': 'error', 'message': friendly})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'error', 'message': f'❌ {err_msg}'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
