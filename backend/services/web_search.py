"""
Web Search Augmentation Service
================================
Provides real-time web search results to supplement the local document RAG pipeline.
Uses DuckDuckGo search (no API key required) for fetching fresh web information.
"""

import os
import json
import re
import urllib.request
import urllib.parse
from typing import List, Dict, Any


def _fetch_url(url: str, timeout: int = 5) -> str:
    """Fetch URL content with a timeout."""
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        import ssl
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as response:
            return response.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"[WebSearch] Failed to fetch {url}: {e}")
        return ""


def _extract_text_from_html(html: str, max_chars: int = 500) -> str:
    """Extract readable text from HTML, stripping tags."""
    # Remove script/style blocks
    html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', html)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_chars]


def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search DuckDuckGo and return results.
    Returns list of dicts with 'title', 'url', 'snippet'.
    """
    results = []
    try:
        # Use DuckDuckGo HTML search (no API key required)
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        html = _fetch_url(url, timeout=8)
        
        if not html:
            return results
        
        # Parse results from DDG HTML
        # Each result is in a div with class "result"
        result_blocks = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )
        
        for link, title, snippet in result_blocks[:max_results]:
            # Clean up the URL (DDG wraps URLs)
            actual_url = link
            if 'uddg=' in link:
                url_match = re.search(r'uddg=([^&]+)', link)
                if url_match:
                    actual_url = urllib.parse.unquote(url_match.group(1))
            
            clean_title = re.sub(r'<[^>]+>', '', title).strip()
            clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
            
            if clean_title and actual_url:
                results.append({
                    'title': clean_title,
                    'url': actual_url,
                    'snippet': clean_snippet
                })
        
    except Exception as e:
        print(f"[WebSearch] DuckDuckGo search failed: {e}")
    
    return results


def fetch_web_context(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Search the web and return formatted context for the LLM.
    Returns dict with 'context_text' (formatted string) and 'sources' (list of source dicts).
    """
    search_results = search_duckduckgo(query, max_results=max_results)
    
    if not search_results:
        return {"context_text": "", "sources": []}
    
    web_sources = []
    context_parts = []
    
    for i, result in enumerate(search_results):
        source = {
            "index": i + 1,
            "title": result['title'],
            "url": result['url'],
            "snippet": result['snippet'],
            "type": "web"
        }
        web_sources.append(source)
        context_parts.append(
            f"Web Source {i+1} ({result['title']}):\n"
            f"URL: {result['url']}\n"
            f"{result['snippet']}"
        )
    
    context_text = "\n\n".join(context_parts)
    
    return {
        "context_text": context_text,
        "sources": web_sources
    }


def should_search_web(query: str, doc_sources: List[Dict]) -> bool:
    """
    Decide whether to augment with web search based on query characteristics
    and the quality of document results.
    
    Returns True if web search would be beneficial.
    """
    query_lower = query.lower()
    
    # Keywords that suggest the user wants current/external information
    web_trigger_words = [
        'latest', 'recent', 'current', 'today', 'news', 'update', 
        'compare', 'versus', 'vs', 'difference between',
        'what is', 'who is', 'how to', 'explain', 'define',
        'best practices', 'industry', 'standard', 'trends',
        '2024', '2025', '2026'
    ]
    
    # Check if query contains web-trigger keywords
    has_web_triggers = any(word in query_lower for word in web_trigger_words)
    
    # Check if document sources have low relevance
    low_doc_relevance = True
    if doc_sources:
        avg_relevance = sum(s.get('relevance_score', 0) for s in doc_sources) / len(doc_sources)
        low_doc_relevance = avg_relevance < 0.35
    
    # Search web if: no documents, low relevance, or query suggests external info needed
    return len(doc_sources) == 0 or low_doc_relevance or has_web_triggers
