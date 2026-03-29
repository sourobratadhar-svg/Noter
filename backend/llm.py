"""
LLM Module — Ollama Local Integration
======================================
Handles all communication with Ollama running locally.
No external network calls — Ollama runs on the same machine.

Usage:
    from llm import OllamaClient
    client = OllamaClient(base_url="http://localhost:11434", model="mistral")
    ok = await client.check_health()
    answer = await client.generate(prompt)
    models = await client.list_models()
"""

import httpx
import re
from typing import List, Optional


class OllamaClient:
    """
    Async wrapper around Ollama's local HTTP API.
    All calls go to localhost — no data leaves the machine.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def check_health(self) -> dict:
        """
        Check if Ollama is running and return detailed status.
        Returns: {"available": bool, "models": [...], "error": str|None}
        """
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    return {
                        "available": True,
                        "models": models,
                        "active_model": self.model,
                        "model_loaded": any(self.model in m for m in models),
                        "error": None
                    }
                return {
                    "available": False, "models": [], "active_model": self.model,
                    "model_loaded": False,
                    "error": f"Ollama returned status {resp.status_code}"
                }
        except httpx.ConnectError:
            return {
                "available": False, "models": [], "active_model": self.model,
                "model_loaded": False,
                "error": "Cannot connect to Ollama. Is it running? Start with: ollama serve"
            }
        except httpx.TimeoutException:
            return {
                "available": False, "models": [], "active_model": self.model,
                "model_loaded": False,
                "error": "Ollama connection timed out. It may be loading a model."
            }
        except Exception as e:
            return {
                "available": False, "models": [], "active_model": self.model,
                "model_loaded": False,
                "error": f"Unexpected error: {str(e)}"
            }

    async def list_models(self) -> List[str]:
        """List all models available in Ollama."""
        status = await self.check_health()
        return status["models"]

    async def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> dict:
        """
        Generate a response from Ollama.
        Returns: {"text": str, "success": bool, "error": str|None, "model": str}
        """
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "text": data.get("response", ""),
                        "success": True,
                        "error": None,
                        "model": self.model,
                        "eval_count": data.get("eval_count", 0),
                        "eval_duration_ms": round(data.get("eval_duration", 0) / 1_000_000, 1),
                    }
                elif resp.status_code == 404:
                    return {
                        "text": "",
                        "success": False,
                        "error": f"Model '{self.model}' not found. Run: ollama pull {self.model}",
                        "model": self.model,
                    }
                else:
                    return {
                        "text": "",
                        "success": False,
                        "error": f"Ollama error (HTTP {resp.status_code}): {resp.text[:200]}",
                        "model": self.model,
                    }
        except httpx.ConnectError:
            return {
                "text": "",
                "success": False,
                "error": "Cannot connect to Ollama. Start with: ollama serve",
                "model": self.model,
            }
        except httpx.TimeoutException:
            return {
                "text": "",
                "success": False,
                "error": "Ollama timed out. The model may be too large or still loading.",
                "model": self.model,
            }
        except Exception as e:
            return {
                "text": "",
                "success": False,
                "error": f"LLM error: {str(e)}",
                "model": self.model,
            }

    def set_model(self, model: str):
        """Change the active model."""
        self.model = model


def extractive_fallback(question: str, contexts: List[str]) -> str:
    """
    Fallback answer generation when Ollama is unavailable.
    Ranks sentences by keyword overlap with the question,
    returning the most relevant excerpts from context.
    Works fully offline — no LLM needed.
    """
    if not contexts:
        return "I couldn't find relevant notes for this question."

    all_text = " ".join(contexts)
    sentences = re.split(r'(?<=[.!?])\s+', all_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return "I couldn't find relevant notes for this question."

    # Score by keyword overlap
    q_words = set(question.lower().split())
    scored = []
    for sent in sentences:
        s_words = set(sent.lower().split())
        overlap = len(q_words & s_words)
        scored.append((overlap, sent))

    scored.sort(key=lambda x: -x[0])
    top_sentences = [s for score, s in scored[:5] if score > 0]
    if not top_sentences:
        return "I couldn't find relevant notes for this question."

    answer = "Based on your notes:\n\n" + " ".join(top_sentences)
    answer += "\n\n[Extractive mode — Ollama offline. Install Ollama for AI-generated answers.]"
    return answer
