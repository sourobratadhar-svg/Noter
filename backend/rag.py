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


def build_rag_prompt(question: str, contexts: List[str], chat_history: List[dict] = None) -> str:
    """
    Construct a grounded prompt for the LLM.
    The prompt instructs the model to answer ONLY from provided context,
    preventing hallucination and ensuring all answers trace to user notes.
    """
    context_block = "\n\n---\n\n".join(contexts)
    
    history_block = ""
    if chat_history:
        history_lines = []
        for msg in chat_history:
            role = "USER" if msg.get("role") == "user" else "ASSISTANT"
            history_lines.append(f"{role}: {msg.get('content', '')}")
        if history_lines:
            history_block = "CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n\n"

    return f"""You are a highly capable personal knowledge assistant. Your core directive is to answer the user's question based strictly on the provided context retrieved from their notes.

{history_block}RELEVANT CONTEXT FROM NOTES:
{context_block}

INSTRUCTIONS:
1. Always prioritize the 'RELEVANT CONTEXT FROM NOTES' over the conversation history to answer the question.
2. Be concise, well-structured, and clear. Use bullet points when helpful.
3. Do not just blindly repeat raw chunks of context—synthesize a fluid answer.
4. If the provided context does not explicitly contain the answer, your ONLY response must be exactly "I couldn't find relevant notes for this question." Do NOT hallucinate or guess.

QUESTION: {question}

ANSWER:"""


class RAGPipeline:
    """
    Complete RAG pipeline: embed → retrieve → prompt → generate.
    Falls back to extractive answers when Ollama is unavailable.
    """

    def __init__(self, embedding_engine: EmbeddingEngine, collection, ollama_client: OllamaClient):
        self.embeddings = embedding_engine
        self.collection = collection
        self.ollama = ollama_client

    async def query(self, question: str, top_k: int = 3, chat_history: List[dict] = None) -> dict:
        """
        Full RAG query pipeline.

        Args:
            question: User's natural language question
            top_k: Number of chunks to retrieve
            chat_history: Optional history of user/assistant interactions

        Returns:
            {
                "answer": str,
                "sources": [{chunk_index, preview, note_title, note_id, relevance}],
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

        # Step 1: Enhance search query conditionally for short follow-ups
        search_query = question
        if len(question.split()) < 4 and chat_history:
            last_user_msg = next((msg.get("content", "") for msg in reversed(chat_history) if msg.get("role") == "user"), None)
            if last_user_msg:
                search_query = f"{last_user_msg} {question}"

        # Step 2: Embed the search query locally
        q_embedding = self.embeddings.encode_single(search_query)

        # Step 3: Retrieve top-k relevant chunks from ChromaDB
        effective_k = min(top_k, total)
        results = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"]
        )

        contexts = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        # Accumulate chunks until a limit of ~4000 characters is reached
        char_limit = 4000
        current_chars = 0
        final_contexts = []
        final_metadatas = []
        final_distances = []
        
        for doc, meta, dist in zip(contexts, metadatas, distances):
            doc_len = len(doc)
            if final_contexts and current_chars + doc_len > char_limit:
                break
            final_contexts.append(doc)
            final_metadatas.append(meta)
            final_distances.append(dist)
            current_chars += doc_len

        # Build source references for frontend
        sources = []
        for i, (doc, meta, dist) in enumerate(zip(final_contexts, final_metadatas, final_distances)):
            preview = doc[:150].replace('\n', ' ').strip() + ("..." if len(doc) > 150 else "")
            sources.append({
                "chunk_index": i,
                "preview": preview,
                "note_title": meta.get("title", "Unknown"),
                "note_id": meta.get("note_id", ""),
                "relevance": round(1 - dist, 3)
            })

        # Step 3: Try Ollama LLM first
        ollama_status = await self.ollama.check_health()
        ollama_error = None

        if ollama_status["available"]:
            # Step 4: Build grounded prompt and generate
            prompt = build_rag_prompt(question, final_contexts, chat_history)
            result = await self.ollama.generate(prompt)

            if result["success"] and result["text"]:
                answer_text = result["text"].strip()
                ollama_sources = sources
                if answer_text == "I couldn't find relevant notes for this question.":
                    ollama_sources = []
                
                return {
                    "answer": answer_text,
                    "sources": ollama_sources,
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
        answer = extractive_fallback(question, final_contexts)
        
        fallback_sources = sources
        if answer.strip() == "I couldn't find relevant notes for this question.":
            fallback_sources = []
            
        return {
            "answer": answer,
            "sources": fallback_sources,
            "ollama_available": False,
            "ollama_error": ollama_error,
            "mode": "extractive",
            "model": self.ollama.model,
        }
