"""
Local LLM Service — Your Own AI, Running on Your Machine
==========================================================
Connects to Ollama (https://ollama.com) which runs open-source LLMs
locally on your computer. No API keys, no cloud, no external services.

Supported models (downloaded automatically by Ollama):
  - phi3        → Microsoft Phi-3 (3.8B params, fast, smart)
  - llama3.2    → Meta Llama 3.2 (3B params, great quality)
  - mistral     → Mistral 7B (powerful, needs more RAM)
  - tinyllama   → TinyLlama (1.1B, ultra-fast, basic)
  - gemma2      → Google Gemma 2 (good balance)

How it works:
  1. Ollama runs as a background service on localhost:11434
  2. This module sends HTTP requests to Ollama's REST API
  3. Responses are streamed back token-by-token (just like cloud LLMs)
  4. All processing happens on YOUR machine — fully private & offline-capable
"""

import os
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Generator, Optional


# ── Configuration ────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # seconds


def is_ollama_running() -> bool:
    """Check if Ollama is running and reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_installed_models() -> List[Dict[str, Any]]:
    """Get list of models installed in Ollama."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = []
            for m in data.get("models", []):
                models.append({
                    "name": m.get("name", ""),
                    "size": m.get("size", 0),
                    "modified_at": m.get("modified_at", ""),
                    "size_gb": round(m.get("size", 0) / (1024**3), 1),
                })
            return models
    except Exception as e:
        print(f"[LocalLLM] Failed to list models: {e}")
        return []


def get_ollama_status() -> Dict[str, Any]:
    """Get full status of the local LLM system."""
    running = is_ollama_running()
    models = get_installed_models() if running else []
    current_model = OLLAMA_MODEL

    # Check if the configured model is installed
    model_names = [m["name"].split(":")[0] for m in models]
    model_ready = any(
        current_model == name or current_model in name
        for name in model_names
    ) if models else False

    return {
        "running": running,
        "url": OLLAMA_BASE_URL,
        "current_model": current_model,
        "model_ready": model_ready,
        "installed_models": models,
        "model_count": len(models),
    }


def pull_model(model_name: str) -> Generator[str, None, None]:
    """
    Pull (download) a model from Ollama's registry.
    Yields progress updates as JSON strings.
    """
    try:
        body = json.dumps({"name": model_name}).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/pull",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    yield json.dumps(chunk)
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        yield json.dumps({"error": str(e)})


def generate_streaming(
    prompt: str,
    model: str = None,
    system: str = None,
    context: List[int] = None,
    temperature: float = 0.7,
) -> Generator[str, None, None]:
    """
    Generate a streaming response from the local LLM.

    Yields text tokens one at a time for real-time display.
    This is the core inference function — runs entirely on your machine.
    """
    model = model or OLLAMA_MODEL

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": 2048,
        },
    }

    if system:
        payload["system"] = system

    if context:
        payload["context"] = context

    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            for line in resp:
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    # Check if generation is done
                    if chunk.get("done", False):
                        return
                except json.JSONDecodeError:
                    continue

    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            f"Make sure Ollama is running (start it with 'ollama serve'). "
            f"Error: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"Local LLM generation failed: {e}")


def chat_streaming(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.7,
) -> Generator[str, None, None]:
    """
    Chat-style generation using Ollama's /api/chat endpoint.
    Supports multi-turn conversation with message history.

    messages format: [{"role": "user"/"assistant"/"system", "content": "..."}]
    Yields text tokens one at a time.
    """
    model = model or OLLAMA_MODEL

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": 2048,
        },
    }

    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            for line in resp:
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    msg = chunk.get("message", {})
                    token = msg.get("content", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        return
                except json.JSONDecodeError:
                    continue

    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            f"Make sure Ollama is running. Error: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"Local LLM chat failed: {e}")
"""
Setup instructions for the user:

1. Download Ollama from: https://ollama.com/download
2. Install it (Windows installer, double-click)
3. It runs automatically as a background service
4. Pull your first model:
     ollama pull phi3
   (or: ollama pull llama3.2, ollama pull mistral)
5. That's it! Your RAGAI app will auto-detect Ollama and use it.
"""
