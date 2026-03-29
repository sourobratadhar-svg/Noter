"""
Privacy-First RAG Notes Server
===============================
All processing happens locally. No external API calls.
Architecture:
  - ChromaDB for persistent vector storage
  - sentence-transformers for local embeddings
  - Ollama for local LLM inference (with extractive fallback)
  - MongoDB for note metadata
"""

from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import chromadb
from sentence_transformers import SentenceTransformer
import httpx
import re

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'rag_notes')]

# ChromaDB persistent storage
CHROMA_PATH = str(ROOT_DIR / "chroma_data")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="notes_vectors",
    metadata={"hnsw:space": "cosine"}
)

# Local embedding model (all-MiniLM-L6-v2, 384 dimensions)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Ollama config
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'mistral')

app = FastAPI()
api_router = APIRouter(prefix="/api")

# ─── Pydantic Models ───

class NoteIngestRequest(BaseModel):
    title: str = ""
    content: str
    source_type: str = "paste"  # paste | file

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

class NoteResponse(BaseModel):
    id: str
    title: str
    source_type: str
    chunk_count: int
    char_count: int
    created_at: str

class AskResponse(BaseModel):
    answer: str
    sources: List[dict]
    ollama_available: bool

class HealthResponse(BaseModel):
    status: str
    chromadb: str
    ollama: str
    embedding_model: str
    total_chunks: int

class StatsResponse(BaseModel):
    total_notes: int
    total_chunks: int
    embedding_model: str
    embedding_dim: int
    ollama_model: str

# ─── Chunking Module ───

def chunk_text(text: str, max_tokens: int = 400, overlap: int = 50) -> List[str]:
    """
    Split text into semantically meaningful chunks.
    Uses paragraph boundaries first, then sentence boundaries,
    then falls back to token-level splitting.
    """
    # Clean the text
    text = text.strip()
    if not text:
        return []

    # Split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # Estimate tokens (~4 chars per token)
        combined = (current_chunk + "\n\n" + para).strip() if current_chunk else para
        estimated_tokens = len(combined) // 4

        if estimated_tokens <= max_tokens:
            current_chunk = combined
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If single paragraph is too long, split by sentences
            if len(para) // 4 > max_tokens:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sub_chunk = ""
                for sent in sentences:
                    combined_sent = (sub_chunk + " " + sent).strip() if sub_chunk else sent
                    if len(combined_sent) // 4 <= max_tokens:
                        sub_chunk = combined_sent
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = sent
                if sub_chunk:
                    current_chunk = sub_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    # Ensure minimum chunk quality
    return [c for c in chunks if len(c) > 20]

# ─── Embedding Module ───

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings locally using sentence-transformers."""
    embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=False)
    return embeddings.tolist()

# ─── Ollama LLM Module ───

async def check_ollama() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client_http:
            resp = await client_http.get(f"{OLLAMA_BASE_URL}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False

async def call_ollama(prompt: str) -> Optional[str]:
    """Call Ollama for LLM generation. Returns None if unavailable."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client_http:
            resp = await client_http.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 512}
                }
            )
            if resp.status_code == 200:
                return resp.json().get("response", "")
    except Exception:
        pass
    return None

def extractive_fallback(question: str, contexts: List[str]) -> str:
    """
    Fallback answer generation when Ollama is unavailable.
    Extracts and ranks relevant sentences from context.
    """
    if not contexts:
        return "No relevant notes found for your question."

    # Combine all contexts
    all_text = " ".join(contexts)
    sentences = re.split(r'(?<=[.!?])\s+', all_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return "Found matching notes but could not extract a clear answer. Here are the relevant passages:\n\n" + "\n".join(contexts[:3])

    # Score sentences by keyword overlap with question
    q_words = set(question.lower().split())
    scored = []
    for sent in sentences:
        s_words = set(sent.lower().split())
        overlap = len(q_words & s_words)
        scored.append((overlap, sent))

    scored.sort(key=lambda x: -x[0])

    # Take top relevant sentences
    top_sentences = [s for _, s in scored[:5] if _ > 0]

    if not top_sentences:
        top_sentences = sentences[:3]

    answer = "Based on your notes:\n\n" + " ".join(top_sentences)
    answer += "\n\n[Note: Ollama is not available. This is an extractive answer from your notes. Install Ollama for AI-generated responses.]"
    return answer

# ─── RAG Pipeline ───

def build_rag_prompt(question: str, contexts: List[str]) -> str:
    """Construct a grounded prompt ensuring answers only use retrieved context."""
    context_block = "\n\n---\n\n".join(contexts)
    return f"""You are a helpful assistant that answers questions based ONLY on the provided context from the user's personal notes. Do not use any external knowledge. If the answer cannot be found in the context, say "I couldn't find relevant information in your notes."

CONTEXT FROM NOTES:
{context_block}

QUESTION: {question}

ANSWER (based strictly on the above context):"""

# ─── API Routes ───

@api_router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check."""
    ollama_ok = await check_ollama()
    total_chunks = collection.count()
    return HealthResponse(
        status="operational",
        chromadb="connected",
        ollama="connected" if ollama_ok else "unavailable",
        embedding_model="all-MiniLM-L6-v2",
        total_chunks=total_chunks
    )

@api_router.post("/ingest")
async def ingest_note(request: NoteIngestRequest):
    """
    Ingest a note: chunk text, generate embeddings, store in ChromaDB + MongoDB.
    """
    content = request.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    note_id = str(uuid.uuid4())
    title = request.title.strip() or f"Note {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"

    # Chunk the text
    chunks = chunk_text(content)
    if not chunks:
        raise HTTPException(status_code=400, detail="Content too short to process")

    # Generate embeddings locally
    embeddings = generate_embeddings(chunks)

    # Store in ChromaDB
    chunk_ids = [f"{note_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"note_id": note_id, "title": title, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    # Store note metadata in MongoDB
    note_doc = {
        "id": note_id,
        "title": title,
        "source_type": request.source_type,
        "chunk_count": len(chunks),
        "char_count": len(content),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.notes.insert_one(note_doc)

    return {
        "id": note_id,
        "title": title,
        "chunk_count": len(chunks),
        "message": f"Ingested {len(chunks)} chunks from note"
    }

@api_router.post("/ingest-file")
async def ingest_file(
    file: UploadFile = File(...),
    title: str = Form("")
):
    """Ingest a .txt or .md file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ("txt", "md"):
        raise HTTPException(status_code=400, detail="Only .txt and .md files supported")

    content_bytes = await file.read()
    content = content_bytes.decode("utf-8", errors="ignore").strip()

    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    # Reuse the ingest logic
    req = NoteIngestRequest(
        title=title or file.filename,
        content=content,
        source_type="file"
    )
    return await ingest_note(req)

@api_router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    RAG query: embed question, retrieve top-k chunks, generate answer.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    total = collection.count()
    if total == 0:
        return AskResponse(
            answer="No notes have been ingested yet. Add some notes first!",
            sources=[],
            ollama_available=False
        )

    # Embed the question
    q_embedding = generate_embeddings([question])[0]

    # Retrieve top-k from ChromaDB
    top_k = min(request.top_k, total)
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    contexts = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    # Build sources for frontend display
    sources = []
    for i, (doc, meta, dist) in enumerate(zip(contexts, metadatas, distances)):
        sources.append({
            "chunk_index": i,
            "text": doc[:300],
            "note_title": meta.get("title", "Unknown"),
            "note_id": meta.get("note_id", ""),
            "relevance": round(1 - dist, 3)  # cosine similarity
        })

    # Try Ollama first, fall back to extractive
    ollama_ok = await check_ollama()
    if ollama_ok:
        prompt = build_rag_prompt(question, contexts)
        answer = await call_ollama(prompt)
        if not answer:
            answer = extractive_fallback(question, contexts)
            ollama_ok = False
    else:
        answer = extractive_fallback(question, contexts)

    return AskResponse(
        answer=answer,
        sources=sources,
        ollama_available=ollama_ok
    )

@api_router.get("/notes", response_model=List[NoteResponse])
async def list_notes():
    """List all ingested notes."""
    notes = await db.notes.find({}, {"_id": 0}).sort("created_at", -1).to_list(1000)
    return [NoteResponse(**n) for n in notes]

@api_router.delete("/notes/{note_id}")
async def delete_note(note_id: str):
    """Delete a note and its vectors from ChromaDB."""
    # Remove from MongoDB
    result = await db.notes.delete_one({"id": note_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Note not found")

    # Remove vectors from ChromaDB
    try:
        # Get all chunk IDs for this note
        all_ids = collection.get(where={"note_id": note_id})["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
    except Exception:
        pass  # ChromaDB cleanup is best-effort

    return {"message": "Note deleted", "id": note_id}

@api_router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    note_count = await db.notes.count_documents({})
    chunk_count = collection.count()
    return StatsResponse(
        total_notes=note_count,
        total_chunks=chunk_count,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dim=384,
        ollama_model=OLLAMA_MODEL
    )

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Minimal logging (no user content logged for privacy)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
