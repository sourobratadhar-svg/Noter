import httpx
import os

class GemmaClient:
    """
    Gemma 4 API integration.
    Implements the same interface as OllamaClient and QwenClient so it can be dynamically swapped.
    """
    def __init__(self, api_key: str = None, base_url: str = "https://openrouter.ai/api/v1", model: str = "google/gemma-4-26b-a4b-it:free"):
        self.api_key = api_key or os.environ.get("GEMMA_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def check_health(self) -> dict:
        """
        Check if Gemma 4 is available.
        If no API key is set, we return available=True but will mock responses
        to allow system testing without breaking.
        """
        return {
            "available": True,
            "models": [self.model],
            "active_model": self.model,
            "model_loaded": True,
            "error": None if self.api_key else "No GEMMA_API_KEY provided. Using mock responses."
        }

    async def list_models(self) -> list:
        return [self.model]

    async def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> dict:
        """
        Generate response from Gemma API. 
        Mocks response if no API key is configured.
        """
        if not self.api_key:
            return {
                "text": f"Gemma 4 (Open Source) is successfully connected! (Mock Response: Set GEMMA_API_KEY in .env to use the real API.)",
                "success": True,
                "error": None,
                "model": self.model,
                "eval_count": 0,
                "eval_duration_ms": 0,
            }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                resp = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return {
                        "text": text,
                        "success": True,
                        "error": None,
                        "model": self.model,
                        "eval_count": data.get("usage", {}).get("completion_tokens", 0),
                        "eval_duration_ms": 0,
                    }
                else:
                    return {
                        "text": "",
                        "success": False,
                        "error": f"Gemma API error ({resp.status_code}): {resp.text[:200]}",
                        "model": self.model,
                    }
        except Exception as e:
            return {
                "text": "",
                "success": False,
                "error": f"Gemma generation error: {str(e)}",
                "model": self.model,
            }

    def set_model(self, model: str):
        self.model = model
