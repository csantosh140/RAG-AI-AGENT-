"""
LLM Engine — Multi-Source Intelligence Layer
==============================================
Orchestrates:
  1. Document retrieval (ChromaDB vector search)
  2. Web search augmentation (DuckDuckGo)
  3. Gemini LLM generation with enhanced prompts
  4. Smart source routing and citation management
  5. Image/Graph/Chart generation via matplotlib

This is the core intelligence layer that makes the AI give accurate,
up-to-date, and well-cited answers from both your datasets AND the web.
"""

import os
import json
import re
import time
import google.generativeai as genai
from typing import List, Dict, Any, AsyncGenerator, Optional
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session

def _extract_image_prompt(query: str) -> str:
    """Create a descriptive image prompt from the user query.
    Removes generic command words but retains any adjectives or context provided by the user.
    If extraction yields a very short term, prepend a detailed description.
    """
    # Lowercase for uniform matching
    lower = query.lower()
    # Remove common leading verbs and keywords
    cleaned = re.sub(r"\b(?:generate|create|make|draw|show|build|design|render|produce|sketch|paint|illustrate|image|photo|picture|illustration|artwork|painting|portrait)\b", "", lower)
    cleaned = cleaned.strip()
    # If the result is too short, assume user gave a simple noun and expand it
    if len(cleaned.split()) <= 2 and cleaned:
        cleaned = f"a highly detailed, photorealistic image of {cleaned}"
    # Capitalize first letter for aesthetic
    return cleaned[:1].upper() + cleaned[1:] if cleaned else query.strip()


from database import ChatMessage, store_chat_message_with_search_data
from services.retrieval import search_documents
from services.web_search import fetch_web_context, should_search_web
from services.image_gen import detect_chart_request, parse_and_generate_chart, HAS_MATPLOTLIB

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

# Try as many models as we have preferred entries
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
    for m in available_models:
        if m not in skip:
            return m
    return None


def _is_quota_error(error):
    """Check if an exception is a 429 quota/rate-limit error."""
    err_str = str(error).lower()
    return '429' in err_str or 'quota' in err_str or 'rate' in err_str or 'resource exhausted' in err_str


# ── Image generation prompt injection ────────────────────────────
IMAGE_GEN_INSTRUCTION = """

## 🖼️ IMAGE/CHART GENERATION CAPABILITY:
You have the ability to generate charts, graphs, visual infographics, and AI photos/images!
When the user asks you to create/generate/draw/visualize/plot/paint/sketch any kind of image, photo, chart, graph, 
diagram, or infographic, you MUST include a special JSON block in your response.

Supported visual/chart types: bar_chart, line_chart, pie_chart, scatter_plot, histogram, comparison, infographic, timeline, ai_photo

When generating a visual, output a JSON block wrapped in ```chart_json ... ``` markers.
The JSON must follow one of these schemas:

**bar_chart**: {"type":"bar_chart", "title":"...", "labels":["A","B"], "values":[10,20], "ylabel":"...", "xlabel":"...", "horizontal": false}
**line_chart**: {"type":"line_chart", "title":"...", "x_data":["Jan","Feb"], "y_datasets":[{"label":"Sales","values":[10,20]}], "xlabel":"Month", "ylabel":"Value"}
**pie_chart**: {"type":"pie_chart", "title":"...", "labels":["A","B"], "values":[60,40]}
**scatter_plot**: {"type":"scatter_plot", "title":"...", "x_data":[1,2,3], "y_data":[4,5,6], "xlabel":"X", "ylabel":"Y"}
**histogram**: {"type":"histogram", "title":"...", "data":[1,2,2,3,3,3,4], "bins":10, "xlabel":"Value", "ylabel":"Frequency"}
**comparison**: {"type":"comparison", "title":"...", "categories":["Q1","Q2"], "group_names":["2023","2024"], "group_values":[[10,20],[15,25]], "ylabel":"Revenue"}
**infographic**: {"type":"infographic", "title":"...", "items":[{"label":"Users","value":"1.2M","icon":"👥"},{"label":"Revenue","value":"$4.5B","icon":"💰"}]}
**timeline**: {"type":"timeline", "title":"...", "events":[{"date":"2020","label":"Founded","desc":"Company established"}]}
**ai_photo**: {"type":"ai_photo", "title":"...", "prompt":"..."}

IMPORTANT RULES for visual elements:
- Use **ai_photo** for realistic photos, paintings, illustrations, drawings, or sketches. The "prompt" field MUST be a detailed, descriptive prompt in English detailing what the visual should show (e.g. "a majestic white dragon on top of a mountain, digital art, highly detailed").
- Always provide realistic, meaningful data in the JSON for charts (use document context or general knowledge)
- Always add a text explanation BEFORE or AFTER the chart/image JSON block
- The data should be accurate based on document context or common knowledge
- Make charts visually informative with clear titles and labels
- If you use document data, cite the source
- You can include MULTIPLE charts or images in a single response if the user asks
"""


def _build_enhanced_prompt(
    query: str,
    doc_sources: List[Dict[str, Any]],
    web_context: Optional[Dict[str, Any]] = None,
    web_search_enabled: bool = False,
    is_image_request: bool = False
) -> str:
    """
    Build an enhanced system prompt that instructs the LLM to combine
    document context with web context for comprehensive answers.
    """
    # Document context
    if doc_sources:
        doc_context = "\n\n".join([
            f"📄 Document Source {s['index']} ({s['filename']}, Page {s.get('page', 1)}):\n{s['content']}"
            for s in doc_sources
        ])
    else:
        doc_context = "(No relevant document sections found.)"
    
    # Web context
    web_text = ""
    if web_context and web_context.get("context_text"):
        web_text = web_context["context_text"]
    
    # Build the system instruction
    system_instruction = (
        "You are an expert AI research assistant with access to the user's uploaded documents "
        "and real-time web information. Your job is to give the MOST ACCURATE and COMPREHENSIVE "
        "answers possible by intelligently combining all available sources.\n\n"
        
        "When responding to technical, computer science, or cybersecurity topics, structure your answers "
        "to focus on educational theories, defensive security practices, standard software engineering guidelines, "
        "and risk mitigation frameworks. Maintain a neutral, educational, and objective tone.\n\n"
        
        "## RULES:\n"
        "1. **Documents First**: Prioritize information from the user's uploaded documents when relevant. "
        "These are the primary knowledge base for document-specific questions.\n"
        "2. **Web Augmentation**: Use web search results to supplement, validate, or add context "
        "to information not covered in the documents.\n"
        "3. **General Knowledge Fallback**: CRITICAL — If the documents and web results do not contain "
        "sufficient information about the user's question, you MUST still answer using your own extensive "
        "general knowledge and training data. NEVER say 'the documents do not contain information about X' "
        "or refuse to answer. Always provide a helpful, complete response.\n"
        "4. **Clear Citations**: Cite your sources when applicable:\n"
        "   - For document sources: use [Doc N] (e.g., [Doc 1], [Doc 2]) and mention the page number.\n"
        "   - For web sources: use [Web N] (e.g., [Web 1]) and mention the source name.\n"
        "   - For general knowledge: simply answer without a citation tag.\n"
        "5. **Accuracy**: If document sources and web sources conflict, note the discrepancy "
        "and explain which is likely more current or authoritative.\n"
        "6. **Completeness**: Synthesize information from all available sources into a coherent, "
        "well-structured answer.\n"
        "7. **Language**: ALWAYS detect the language of the user's question and respond in that "
        "EXACT same language (e.g., Kannada, Hindi, Telugu, English).\n"
        "8. **Never Refuse**: You MUST always provide a useful, informative answer. "
        "If topic is not in documents, answer from general knowledge and mention that. "
        "Do NOT output error messages or say the information is unavailable.\n\n"
        
        "## FORMAT:\n"
        "- Use markdown formatting for readability (headings, bold, lists, etc.)\n"
        "- Start with a direct answer, then provide supporting details\n"
        "- End with a brief source summary if multiple sources were used\n"
    )
    
    # Add image generation instructions always so the model knows it can generate charts & photos
    system_instruction += IMAGE_GEN_INSTRUCTION
    
    # Assemble the full prompt
    prompt_parts = [system_instruction]
    
    prompt_parts.append(f"\n\n## UPLOADED DOCUMENT CONTEXT:\n{doc_context}")
    
    if web_text:
        prompt_parts.append(f"\n\n## WEB SEARCH RESULTS:\n{web_text}")
    elif web_search_enabled:
        prompt_parts.append("\n\n## WEB SEARCH RESULTS:\n(No relevant web results found.)")
    
    prompt_parts.append(f"\n\n## USER QUESTION:\n{query}")
    
    if is_image_request:
        prompt_parts.append("\n\n**IMPORTANT: The user is specifically requesting a visual/chart/graph. "
                           "You MUST include a ```chart_json block with the appropriate chart data. "
                           "Choose the most suitable chart type for their request.**")
    
    return "\n".join(prompt_parts)

VALID_CHART_TYPES = {'bar_chart', 'line_chart', 'pie_chart', 'scatter_plot', 'histogram', 'comparison', 'infographic', 'timeline', 'ai_photo'}

def _extract_chart_specs(text: str) -> List[Dict[str, Any]]:
    """
    Extract chart JSON specs from the LLM response text.
    Handles multiple formats:
      1. ```chart_json { ... } ```
      2. ```json { ... } ```
      3. ``` { ... } ```
      4. Raw JSON blocks with "type" matching a known chart type
    Returns list of parsed chart specifications.
    """
    charts = []
    
    # Pattern 1: ```chart_json ... ```
    for match in re.findall(r'```chart_json\s*\n?(.*?)```', text, re.DOTALL):
        _try_parse_chart(match.strip(), charts)
    
    # Pattern 2: ```json ... ``` containing chart type
    for match in re.findall(r'```json\s*\n?(.*?)```', text, re.DOTALL):
        _try_parse_chart(match.strip(), charts)
    
    # Pattern 3: ``` ... ``` generic code blocks containing chart JSON
    if not charts:
        for match in re.findall(r'```\s*\n?(.*?)```', text, re.DOTALL):
            _try_parse_chart(match.strip(), charts)
    
    # Pattern 4: Raw JSON objects with "type" field (no code block wrapper)
    if not charts:
        # Find JSON-like objects in the text
        for match in re.finditer(r'\{[^{}]*"type"\s*:\s*"(\w+)"[^{}]*\}', text, re.DOTALL):
            json_str = match.group(0)
            _try_parse_chart(json_str, charts)
        
        # Also try multi-line JSON objects (with nested arrays/objects)
        for match in re.finditer(r'\{\s*\n\s*"type"\s*:.*?\n\}', text, re.DOTALL):
            _try_parse_chart(match.group(0), charts)
    
    return charts


def _try_parse_chart(json_str: str, charts: list):
    """Attempt to parse a string as a chart spec and add to charts list if valid."""
    try:
        spec = json.loads(json_str)
        if isinstance(spec, dict) and spec.get('type') in VALID_CHART_TYPES:
            # Avoid duplicates
            if not any(c.get('title') == spec.get('title') and c.get('type') == spec.get('type') for c in charts):
                charts.append(spec)
    except (json.JSONDecodeError, ValueError):
        pass


def _clean_chart_blocks(text: str) -> str:
    """
    Remove chart JSON blocks from text for clean display.
    Handles all formats: ```chart_json, ```json, ```, and raw JSON.
    """
    # Remove ```chart_json ... ``` blocks
    def _replace_chart_json(match):
        content = match.group(1)
        try:
            spec = json.loads(content.strip())
            if isinstance(spec, dict) and spec.get('type') == 'ai_photo':
                return '\n\n🎨 *[Image generated — see below]*\n\n'
        except Exception:
            pass
        return '\n\n📊 *[Chart generated — see below]*\n\n'

    text = re.sub(r'```chart_json\s*\n?(.*?)```', _replace_chart_json, text, flags=re.DOTALL)
    
    # Remove ```json ... ``` blocks that contain chart types
    def _replace_json_block(match):
        content = match.group(1)
        try:
            spec = json.loads(content.strip())
            if isinstance(spec, dict) and spec.get('type') in VALID_CHART_TYPES:
                if spec.get('type') == 'ai_photo':
                    return '\n\n🎨 *[Image generated — see below]*\n\n'
                return '\n\n📊 *[Chart generated — see below]*\n\n'
        except (json.JSONDecodeError, ValueError):
            pass
        return match.group(0)  # Keep non-chart JSON blocks
    
    text = re.sub(r'```json\s*\n?(.*?)```', _replace_json_block, text, flags=re.DOTALL)
    text = re.sub(r'```\s*\n?(\{.*?"type"\s*:.*?\})```', _replace_json_block, text, flags=re.DOTALL)
    
    # Remove raw JSON blocks that look like chart specs
    for chart_type in VALID_CHART_TYPES:
        placeholder = '\n\n🎨 *[Image generated — see below]*\n\n' if chart_type == 'ai_photo' else '\n\n📊 *[Chart generated — see below]*\n\n'
        # Multi-line raw JSON
        text = re.sub(
            r'\{\s*\n\s*"type"\s*:\s*"' + chart_type + r'".*?\n\}',
            placeholder,
            text, flags=re.DOTALL
        )
        # Single-line raw JSON
        text = re.sub(
            r'\{[^{}]*"type"\s*:\s*"' + chart_type + r'"[^{}]*\}',
            placeholder.strip(),
            text
        )
    
    return text


async def generate_enhanced_response(
    query: str,
    session_id: int,
    db: Session,
    history: List[ChatMessage],
    web_search_enabled: bool = True
) -> AsyncGenerator[str, None]:
    """
    Enhanced response generation that combines:
    1. Document retrieval from ChromaDB
    2. Optional web search augmentation
    3. Gemini LLM with intelligent prompting
    4. Image/Chart/Graph generation via matplotlib
    
    Streams SSE events for real-time frontend rendering.
    """
    
    # ── Step 0a: Parse attached files from query if present ──────
    attached_files = []
    clean_query = query
    
    # Format of attach tag: [AttachedFiles: file1.pdf, file2.docx]
    match = re.match(r'^\[AttachedFiles:\s*(.*?)\s*\]\s*(.*)$', query, re.DOTALL)
    if match:
        files_str = match.group(1).strip()
        clean_query = match.group(2).strip()
        if files_str:
            attached_files = [f.strip() for f in files_str.split(',') if f.strip()]
            
    # ── Step 0: Detect if this is an image/chart request ─────────
    chart_type_hint = detect_chart_request(clean_query)
    is_image_request = chart_type_hint is not None
    is_photo_request = chart_type_hint == 'ai_photo'
    
    # ── Step 0b: For AI photo requests, generate IMMEDIATELY ─────
    if is_photo_request:
        yield f"data: {json.dumps({'type': 'status', 'message': '🎨 Generating AI image...'})}\n\n"
        
        # Extract a concise image description from the user's request
        from services.image_gen import generate_ai_photo
        image_prompt = clean_query

        image_id = generate_ai_photo(image_prompt)
        
        if image_id:
            yield f"data: {json.dumps({'type': 'image', 'image_id': image_id, 'chart_type': 'ai_photo', 'title': clean_query[:80]})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': f'Here is the generated image for: **{clean_query}**'})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'token', 'content': '⚠ Sorry, I could not generate the image. The AI image service may be temporarily unavailable. Please try again.'})}\n\n"
        
        # Save to DB
        response_text = f"[Generated AI image for: {clean_query}]"
        sources_data = {
            "doc_sources": [],
            "web_sources": [],
            "generated_images": [{"image_id": image_id, "chart_type": "ai_photo", "title": clean_query[:80]}] if image_id else []
        }
        store_chat_message_with_search_data(
            db=db,
            session_id=session_id,
            role="assistant",
            content=response_text,
            search_sources=sources_data
        )
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return
    
    if is_image_request:
        yield f"data: {json.dumps({'type': 'status', 'message': '📊 Preparing chart generation...'})}\n\n"
    
    # ── Step 1: Retrieve document context ────────────────────────
    doc_sources = search_documents(clean_query, filenames=attached_files)
    
    # ── Step 2: Yield document sources to frontend ───────────────
    yield f"data: {json.dumps({'type': 'sources', 'sources': doc_sources})}\n\n"
    
    # ── Step 3: Optionally search the web ────────────────────────
    web_context = None
    web_sources = []
    
    if web_search_enabled and should_search_web(clean_query, doc_sources):
        # Notify frontend that web search is happening
        yield f"data: {json.dumps({'type': 'status', 'message': '🌐 Searching the web for updated information...'})}\n\n"
        
        try:
            web_context = fetch_web_context(clean_query, max_results=3)
            web_sources = web_context.get("sources", [])
            
            if web_sources:
                yield f"data: {json.dumps({'type': 'web_sources', 'sources': web_sources})}\n\n"
        except Exception as e:
            print(f"[LLM Engine] Web search failed (non-critical): {e}")
            # Continue without web results — not a fatal error
    
    # ── Step 4: Build enhanced prompt ────────────────────────────
    prompt = _build_enhanced_prompt(clean_query, doc_sources, web_context, web_search_enabled, is_image_request)
    
    if not GEMINI_API_KEY:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Gemini API key is missing. Please set it in .env'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return
    
    # ── Step 5: Stream LLM response with retry logic ────────────
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
                
                # Convert history (excluding the current user message at the end, which is sent as the prompt)
                chat_history = []
                prev_messages = history[:-1] if history else []
                for msg in prev_messages:
                    if not msg.content or not msg.content.strip():
                        continue
                    role = "user" if msg.role == "user" else "model"
                    if chat_history and chat_history[-1]["role"] == role:
                        chat_history[-1]["parts"][0] += "\n" + msg.content
                    else:
                        chat_history.append({"role": role, "parts": [msg.content]})
                
                chat = model.start_chat(history=chat_history)
                response = chat.send_message(prompt, stream=True)
                
                full_response = ""
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.text})}\n\n"
                
                # ── Step 6: Process charts/images from response ──────
                chart_specs = _extract_chart_specs(full_response)
                generated_images_list = []
                if chart_specs:
                    yield f"data: {json.dumps({'type': 'status', 'message': '📊 Generating visualizations...'})}\n\n"
                    
                    for spec in chart_specs:
                        chart_type = spec.get('type', 'bar_chart')
                        
                        # Only skip if it's a matplotlib chart and matplotlib is missing
                        if chart_type != 'ai_photo' and not HAS_MATPLOTLIB:
                            print(f"[ImageGen] Skipping {chart_type} generation because matplotlib is unavailable.")
                            continue
                            
                        image_id = parse_and_generate_chart(chart_type, spec)
                        
                        if image_id:
                            yield f"data: {json.dumps({'type': 'image', 'image_id': image_id, 'chart_type': chart_type, 'title': spec.get('title', 'Generated Image')})}\n\n"
                            generated_images_list.append({
                                "image_id": image_id,
                                "chart_type": chart_type,
                                "title": spec.get('title', 'Generated Image')
                            })
                            print(f"[ImageGen] Generated {chart_type}: {image_id}")
                
                # Save assistant message to DB (clean version without chart JSON)
                clean_response = _clean_chart_blocks(full_response) if chart_specs else full_response
                
                # Store all search sources & context in database via helper function
                sources_data = {
                    "doc_sources": doc_sources,
                    "web_sources": web_sources,
                    "generated_images": generated_images_list
                }
                web_query_text = web_context.get("query") if (web_context and isinstance(web_context, dict)) else None
                
                store_chat_message_with_search_data(
                    db=db,
                    session_id=session_id,
                    role="assistant",
                    content=clean_response,
                    search_sources=sources_data,
                    web_search_query=web_query_text
                )
                
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return  # Success
                
            except Exception as model_err:
                last_error = model_err
                if _is_quota_error(model_err):
                    skip_models.add(selected_model)
                    model_name = selected_model.split('/')[-1]
                    print(f"[Retry {attempt+1}/{MAX_RETRIES}] Quota hit on {model_name}, trying next model...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise model_err
        
        raise last_error or ValueError("__quota_exhausted__")
        
    except Exception as e:
        err_msg = str(e)
        # Replace raw 429/quota API errors with a friendly message
        if '__quota_exhausted__' in err_msg or _is_quota_error(e):
            friendly = (
                "⚠️ **All Gemini models are temporarily at capacity** (free-tier quota reached).\n\n"
                "Please try one of the following:\n"
                "- Wait a minute and ask again\n"
                "- Add a paid Gemini API key to your `.env` file\n"
                "- Visit [Google AI Studio](https://aistudio.google.com) to check your quota"
            )
            yield f"data: {json.dumps({'type': 'error', 'message': friendly})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'error', 'message': f'❌ {err_msg}'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
