import os
import json
import fitz  # PyMuPDF
from datetime import datetime
import google.generativeai as genai
from typing import Dict, Any

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

PREFERRED_MODELS = [
    'models/gemini-2.0-flash',
    'models/gemini-1.5-flash',
    'models/gemini-1.5-pro',
]

def _get_model():
    """Pick the best available Gemini model."""
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key is not configured.")
    try:
        available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in PREFERRED_MODELS:
            if m in available:
                return genai.GenerativeModel(m)
        if available:
            return genai.GenerativeModel(available[0])
    except Exception as e:
        print(f"Error picking model: {e}")
    # Fallback to standard name
    return genai.GenerativeModel('models/gemini-2.0-flash')

def generate_topic_content(topic: str) -> Dict[str, Any]:
    """Use Gemini to generate structured JSON content for the requested topic."""
    model = _get_model()
    
    prompt = f"""
You are an expert researcher and technical writer. Create a comprehensive, high-quality, and structured reference document on the topic: "{topic}".
The document must be informative, detailed, and professional.

Return the content as a single JSON object. Do not output any markdown wrapper or backticks, just raw JSON. The JSON must follow this exact structure:
{{
  "title": "A professional and engaging title",
  "subtitle": "An informative subtitle summarizing the scope of the document",
  "summary": "A detailed executive summary paragraph of the topic.",
  "sections": [
    {{
      "heading": "First Section Heading (e.g. Introduction & Background)",
      "content": "Paragraph 1 explaining the background.\\n\\nParagraph 2 with more details."
    }},
    {{
      "heading": "Second Section Heading (e.g. Core Concepts & Architecture)",
      "content": "Detailed explanation of core concepts, processes, or technologies involved."
    }},
    {{
      "heading": "Third Section Heading (e.g. Key Benefits & Challenges)",
      "content": "Discussion of pros, cons, use-cases, benefits, and common challenges."
    }},
    {{
      "heading": "Fourth Section Heading (e.g. Future Outlook & Trends)",
      "content": "Future expectations, next-generation developments, and emerging trends."
    }}
  ],
  "conclusion": "A solid closing section summarizing key takeaways."
}}
"""
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean up any potential markdown code block wrappers
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        data = json.loads(text)
        return data
    except Exception as e:
        print(f"Error generating topic content via LLM: {e}")
        # Fallback structured content in case LLM fails
        return {
            "title": f"Report on {topic}",
            "subtitle": "Overview and Key Insights",
            "summary": f"This document provides an overview of {topic}, detailing its main components and applications.",
            "sections": [
                {
                    "heading": "1. Introduction",
                    "content": f"The subject of {topic} plays a significant role in modern fields. This section covers its general definitions and scope."
                },
                {
                    "heading": "2. Main Features",
                    "content": "Understanding the core attributes and parameters is critical to applying these concepts effectively."
                }
            ],
            "conclusion": "In conclusion, this topic remains highly relevant and is expected to evolve with future developments."
        }

def wrap_text(text: str, width: float, fontname: str, fontsize: float) -> list[str]:
    """Helper to wrap text to fit within a specific pixel width using PyMuPDF measurements."""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = " ".join(current_line + [word])
        w = fitz.get_text_length(test_line, fontname=fontname, fontsize=fontsize)
        if w <= width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                lines.append(word)
    if current_line:
        lines.append(" ".join(current_line))
    return lines

def generate_pdf_from_data(data: Dict[str, Any]) -> bytes:
    """Generate a styled PDF from the structured data using PyMuPDF."""
    doc = fitz.open()
    
    # ── Style Tokens (Theme: Premium Deep Slate / Blue) ──
    color_primary = (26/255, 54/255, 93/255)     # Deep Blue
    color_secondary = (49/255, 130/255, 206/255) # Light Accent Blue
    color_text_dark = (45/255, 55/255, 72/255)   # Charcoal
    color_text_light = (113/255, 128/255, 150/255) # Slate Gray
    
    font_bold = "helv-bold"
    font_regular = "helv"
    font_oblique = "helv-oblique"
    
    # Page setup
    rect_a4 = fitz.PaperSize("A4")
    page_width, page_height = rect_a4
    margin = 54.0 # 0.75 inch
    content_width = page_width - (margin * 2)
    
    # ── 1. COVER PAGE ──
    cover = doc.new_page()
    
    # Elegant left border decoration
    cover.draw_rect(
        fitz.Rect(0, 0, 18, page_height),
        color=color_primary,
        fill=color_primary
    )
    
    # Accent color badge
    cover.draw_rect(
        fitz.Rect(54, 150, 150, 154),
        color=color_secondary,
        fill=color_secondary
    )
    
    # Title
    title = data.get("title", "Untitled Document")
    title_wrapped = wrap_text(title, content_width - 20, font_bold, 28)
    y = 200
    for line in title_wrapped:
        cover.insert_text(fitz.Point(54, y), line, fontname=font_bold, fontsize=28, color=color_primary)
        y += 36
        
    # Subtitle
    subtitle = data.get("subtitle", "")
    subtitle_wrapped = wrap_text(subtitle, content_width - 20, font_oblique, 14)
    y += 15
    for line in subtitle_wrapped:
        cover.insert_text(fitz.Point(54, y), line, fontname=font_oblique, fontsize=14, color=color_secondary)
        y += 20
        
    # Metadata footer
    cover.draw_line(fitz.Point(54, page_height - 120), fitz.Point(page_width - 54, page_height - 120), color=color_text_light, width=0.5)
    cover.insert_text(fitz.Point(54, page_height - 100), "RAG AI Agent Reference Library", fontname=font_bold, fontsize=10, color=color_text_light)
    date_str = datetime.now().strftime("%B %d, %Y")
    cover.insert_text(fitz.Point(54, page_height - 85), f"Generated on {date_str} · Technical Content", fontname=font_regular, fontsize=9, color=color_text_light)
    
    # ── 2. CONTENT PAGES ──
    page = None
    y = 0
    
    def start_new_content_page():
        nonlocal page, y
        page = doc.new_page()
        # Draw header
        page.insert_text(fitz.Point(margin, 40), title[:60] + ("..." if len(title) > 60 else ""), fontname=font_oblique, fontsize=8, color=color_text_light)
        page.draw_line(fitz.Point(margin, 46), fitz.Point(page_width - margin, 46), color=color_text_light, width=0.5)
        # Reset y
        y = 75
        
    start_new_content_page()
    
    # Helper to check page bounds and create new page if necessary
    def check_page_space(required_height):
        nonlocal page, y
        if y + required_height > page_height - margin:
            # Draw page number on old page before creating new page
            page_num = doc.page_count - 1 # excluding cover as page 1, or just using global page count
            page.insert_text(fitz.Point(page_width / 2 - 10, page_height - 35), str(page_num + 1), fontname=font_regular, fontsize=9, color=color_text_light)
            start_new_content_page()
            
    # --- RENDER EXECUTIVE SUMMARY ---
    summary = data.get("summary", "")
    if summary:
        check_page_space(40)
        page.insert_text(fitz.Point(margin, y), "Executive Summary", fontname=font_bold, fontsize=14, color=color_primary)
        y += 22
        
        paragraphs = summary.split("\n\n")
        for para in paragraphs:
            lines = wrap_text(para.strip(), content_width, font_regular, 10.5)
            for line in lines:
                check_page_space(15)
                page.insert_text(fitz.Point(margin, y), line, fontname=font_regular, fontsize=10.5, color=color_text_dark)
                y += 14.5
            y += 8 # spacing between paragraphs
        y += 15 # spacing after summary section
        
    # --- RENDER SECTIONS ---
    sections = data.get("sections", [])
    for sec in sections:
        heading = sec.get("heading", "")
        content = sec.get("content", "")
        
        # Render heading
        if heading:
            check_page_space(35)
            page.insert_text(fitz.Point(margin, y), heading, fontname=font_bold, fontsize=13, color=color_primary)
            y += 20
            
        # Render paragraphs
        paragraphs = content.split("\n\n")
        for para in paragraphs:
            lines = wrap_text(para.strip(), content_width, font_regular, 10.5)
            for line in lines:
                check_page_space(15)
                page.insert_text(fitz.Point(margin, y), line, fontname=font_regular, fontsize=10.5, color=color_text_dark)
                y += 14.5
            y += 8 # spacing between paragraphs
        y += 15 # spacing after section
        
    # --- RENDER CONCLUSION ---
    conclusion = data.get("conclusion", "")
    if conclusion:
        check_page_space(35)
        page.insert_text(fitz.Point(margin, y), "Conclusion", fontname=font_bold, fontsize=13, color=color_primary)
        y += 20
        
        paragraphs = conclusion.split("\n\n")
        for para in paragraphs:
            lines = wrap_text(para.strip(), content_width, font_regular, 10.5)
            for line in lines:
                check_page_space(15)
                page.insert_text(fitz.Point(margin, y), line, fontname=font_regular, fontsize=10.5, color=color_text_dark)
                y += 14.5
            y += 8
            
    # Add page number to the final page
    page_num = doc.page_count - 1
    page.insert_text(fitz.Point(page_width / 2 - 10, page_height - 35), str(page_num + 1), fontname=font_regular, fontsize=9, color=color_text_light)
    
    # Save to bytes
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes

def generate_pdf_on_topic(topic: str) -> tuple[bytes, str]:
    """Generates structured content and compiles it into a styled PDF.
    
    Returns:
        (pdf_bytes, filename)
    """
    # 1. Get LLM content
    data = generate_topic_content(topic)
    
    # 2. Render to PDF
    pdf_bytes = generate_pdf_from_data(data)
    
    # 3. Create a clean filename
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '_', '-')).strip()
    safe_topic = safe_topic.replace(' ', '_')[:50]
    filename = f"Topic_Report_{safe_topic}.pdf"
    
    return pdf_bytes, filename
