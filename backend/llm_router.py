class LLMRouter:
    """
    Routes generation requests to the active LLM engine.
    Ensures seamless switching between Ollama and Qwen without breaking the RAG pipeline.
    """
    def __init__(self, ollama_client, qwen_client):
        self.ollama = ollama_client
        self.qwen = qwen_client
        self.active_engine = "qwen"

    def set_engine(self, engine: str) -> bool:
        if engine in ["ollama", "qwen"]:
            self.active_engine = engine
            return True
        return False

    def get_active_client(self):
        return self.qwen if self.active_engine == "qwen" else self.ollama

    @property
    def model(self):
        return self.get_active_client().model

    async def check_health(self) -> dict:
        return await self.get_active_client().check_health()

    async def list_models(self) -> list:
        return await self.get_active_client().list_models()

    async def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> dict:
        return await self.get_active_client().generate(prompt, temperature, max_tokens)
