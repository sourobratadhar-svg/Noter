"""
RAG Pipeline Module — Retrieval-Augmented Generation
=====================================================
Orchestrates the full RAG flow:
  1. Embed user question locally
  2. Retrieve top-k chunks from ChromaDB by cosine similarity
  3. Build grounded prompt with context
  4. Generate answer via Ollama (or extractive fallback)

All processing local — no data leaves the machine.

Usage:
    from rag import RAGPipeline
    pipeline = RAGPipeline(embeddings, collection, ollama_client)
    result = await pipeline.query("What is deep learning?")
"""

from typing import List
from llm import OllamaClient, extractive_fallback
from embeddings import EmbeddingEngine


def build_rag_prompt(question: str, contexts: List[str]) -> str:
    """
    Construct a grounded prompt for the LLM.
    The prompt instructs the model to answer ONLY from provided context,
    preventing hallucination and ensuring all answers trace to user notes.
    """
    context_block = "\n\n---\n\n".join(contexts)
    return f"""You are a helpful assistant that answers questions based ONLY on the provided context from the user's personal notes. Do not use any external knowledge. If the answer cannot be found in the context, say "I couldn't find relevant information in your notes."

Be concise and direct. Quote relevant parts of the notes when helpful.

CONTEXT FROM NOTES:
{context_block}

QUESTION: {question}

ANSWER (based strictly on the above context):"""


class RAGPipeline:
    """
    Complete RAG pipeline: embed → retrieve → prompt → generate.
    Falls back to extractive answers when Ollama is unavailable.
    """

    def __init__(self, embedding_engine: EmbeddingEngine, collection, ollama_client: OllamaClient):
        self.embeddings = embedding_engine
        self.collection = collection
        self.ollama = ollama_client

    async def query(self, question: str, top_k: int = 5) -> dict:
        """
        Full RAG query pipeline.

        Args:
            question: User's natural language question
            top_k: Number of chunks to retrieve (default 5, capped to collection size)

        Returns:
            {
                "answer": str,
                "sources": [{chunk_index, text, note_title, note_id, relevance}],
                "ollama_available": bool,
                "ollama_error": str|None,
                "mode": "ollama" | "extractive",
                "model": str
            }
        """
        total = self.collection.count()
        if total == 0:
            return {
                "answer": "No notes have been ingested yet. Add some notes first!",
                "sources": [],
                "ollama_available": False,
                "ollama_error": None,
                "mode": "none",
                "model": self.ollama.model,
            }

        # Step 1: Embed the question locally
        q_embedding = self.embeddings.encode_single(question)

        # Step 2: Retrieve top-k relevant chunks from ChromaDB
        effective_k = min(top_k, total)
        results = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"]
        )

        contexts = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        # Build source references for frontend
        sources = []
        for i, (doc, meta, dist) in enumerate(zip(contexts, metadatas, distances)):
            sources.append({
                "chunk_index": i,
                "text": doc[:300],
                "note_title": meta.get("title", "Unknown"),
                "note_id": meta.get("note_id", ""),
                "relevance": round(1 - dist, 3)
            })

        # Step 3: Try Ollama LLM first
        ollama_status = await self.ollama.check_health()
        ollama_error = None

        if ollama_status["available"]:
            # Step 4: Build grounded prompt and generate
            prompt = build_rag_prompt(question, contexts)
            result = await self.ollama.generate(prompt)

            if result["success"] and result["text"]:
                return {
                    "answer": result["text"],
                    "sources": sources,
                    "ollama_available": True,
                    "ollama_error": None,
                    "mode": "ollama",
                    "model": result["model"],
                }
            else:
                ollama_error = result.get("error", "Generation failed")

        else:
            ollama_error = ollama_status.get("error", "Ollama not available")

        # Step 5: Fallback to extractive answer
        answer = extractive_fallback(question, contexts)
        return {
            "answer": answer,
            "sources": sources,
            "ollama_available": False,
            "ollama_error": ollama_error,
            "mode": "extractive",
            "model": self.ollama.model,
        }
