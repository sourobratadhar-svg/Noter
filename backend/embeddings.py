"""
Embeddings Module — Local Sentence-Transformers
================================================
Generates vector embeddings locally using sentence-transformers.
No network calls — model runs entirely on-device.
Model: all-MiniLM-L6-v2 (384-dim, ~80MB)

Usage:
    from embeddings import EmbeddingEngine
    engine = EmbeddingEngine()
    vectors = engine.encode(["hello world", "test query"])
"""

from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingEngine:
    """
    Local embedding engine. Downloads model on first use,
    then runs entirely offline. No API keys needed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        Batched for efficiency — processes 32 texts at a time.
        Returns list of float vectors.
        """
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return embeddings.tolist()

    def encode_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.encode([text])[0]

    @property
    def info(self) -> dict:
        """Return model metadata."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
        }
