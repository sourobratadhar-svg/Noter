"""
Privacy-First RAG Notes Server
===============================
All processing happens locally. No external API calls.

Architecture:
  - ChromaDB for persistent vector storage
  - sentence-transformers for local embeddings (all-MiniLM-L6-v2)
  - Ollama for local LLM inference (with extractive fallback)
  - MongoDB for note metadata + graph cache

Modules:
  - Ingestion: chunk_text() — semantic paragraph/sentence splitting
  - Embeddings: generate_embeddings() — batched local encoding
  - Retrieval: RAG pipeline with top-k cosine search
  - Graph: compute_knowledge_graph() — pairwise similarity edges
  - LLM: Ollama wrapper with extractive fallback
"""

from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException, Query
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
import numpy as np
import hashlib

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# ─── Database Connections ───

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'rag_notes')]

# ChromaDB persistent storage (local vector database)
CHROMA_PATH = str(ROOT_DIR / "chroma_data")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="notes_vectors",
    metadata={"hnsw:space": "cosine"}
)

# Local embedding model — runs entirely on-device, 384-dim vectors
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Ollama config (local LLM — no network calls)
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'mistral')

# Graph similarity threshold — edges created for chunk pairs above this
GRAPH_SIMILARITY_THRESHOLD = float(os.environ.get('GRAPH_SIMILARITY_THRESHOLD', '0.65'))

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

class PaginatedNotesResponse(BaseModel):
    notes: List[NoteResponse]
    total: int
    page: int
    limit: int
    total_pages: int

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

class GraphNode(BaseModel):
    id: str
    label: str
    note_title: str
    note_id: str
    chunk_index: int
    text_preview: str

class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float

class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    threshold: float
    cached: bool

# ─── Ingestion Module: Chunking ───

def chunk_text(text: str, max_tokens: int = 400, overlap: int = 50) -> List[str]:
    """
    Split text into semantically meaningful chunks (300-500 token range).
    Strategy: paragraph boundaries → sentence boundaries → hard split.
    Tokens estimated at ~4 chars each.
    """
    text = text.strip()
    if not text:
        return []

    # Split by double-newlines (paragraph boundaries)
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        combined = (current_chunk + "\n\n" + para).strip() if current_chunk else para
        estimated_tokens = len(combined) // 4

        if estimated_tokens <= max_tokens:
            current_chunk = combined
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Long paragraph: split by sentences
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

    # Filter out tiny fragments that lack semantic value
    return [c for c in chunks if len(c) > 20]

# ─── Embeddings Module ───

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings locally using sentence-transformers.
    Batched for efficiency — no network calls made.
    """
    embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=False)
    return embeddings.tolist()

# ─── LLM Module: Ollama ───

async def check_ollama() -> bool:
    """Check if Ollama is running locally."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client_http:
            resp = await client_http.get(f"{OLLAMA_BASE_URL}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False

async def call_ollama(prompt: str) -> Optional[str]:
    """Call local Ollama for LLM generation. Returns None if unavailable."""
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
    Ranks sentences by keyword overlap with the question,
    returning the most relevant excerpts from context.
    """
    if not contexts:
        return "No relevant notes found for your question."

    all_text = " ".join(contexts)
    sentences = re.split(r'(?<=[.!?])\s+', all_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return "Found matching notes but could not extract a clear answer. Here are the relevant passages:\n\n" + "\n".join(contexts[:3])

    # Score by keyword overlap (simple but effective for extractive)
    q_words = set(question.lower().split())
    scored = []
    for sent in sentences:
        s_words = set(sent.lower().split())
        overlap = len(q_words & s_words)
        scored.append((overlap, sent))

    scored.sort(key=lambda x: -x[0])
    top_sentences = [s for score, s in scored[:5] if score > 0]

    if not top_sentences:
        top_sentences = sentences[:3]

    answer = "Based on your notes:\n\n" + " ".join(top_sentences)
    answer += "\n\n[Note: Ollama is not available. This is an extractive answer from your notes. Install Ollama for AI-generated responses.]"
    return answer

# ─── RAG Retrieval Pipeline ───

def build_rag_prompt(question: str, contexts: List[str]) -> str:
    """Construct a grounded prompt — answers must come strictly from context."""
    context_block = "\n\n---\n\n".join(contexts)
    return f"""You are a helpful assistant that answers questions based ONLY on the provided context from the user's personal notes. Do not use any external knowledge. If the answer cannot be found in the context, say "I couldn't find relevant information in your notes."

CONTEXT FROM NOTES:
{context_block}

QUESTION: {question}

ANSWER (based strictly on the above context):"""

# ─── Knowledge Graph Module ───

def _compute_collection_hash() -> str:
    """
    Generate a hash representing the current state of the ChromaDB collection.
    Used to invalidate the graph cache when chunks change.
    """
    total = collection.count()
    if total == 0:
        return "empty"
    # Use count + first/last IDs as a fast fingerprint
    all_data = collection.get(limit=1, include=[])
    first_id = all_data["ids"][0] if all_data["ids"] else ""
    return hashlib.md5(f"{total}:{first_id}".encode()).hexdigest()

async def get_cached_graph(collection_hash: str, threshold: float):
    """Retrieve cached graph from MongoDB if still valid."""
    cache = await db.graph_cache.find_one(
        {"collection_hash": collection_hash, "threshold": threshold},
        {"_id": 0}
    )
    return cache

async def save_graph_cache(collection_hash: str, threshold: float, nodes: list, edges: list):
    """Save computed graph to MongoDB for fast retrieval."""
    await db.graph_cache.delete_many({"threshold": threshold})
    await db.graph_cache.insert_one({
        "collection_hash": collection_hash,
        "threshold": threshold,
        "nodes": nodes,
        "edges": edges,
        "computed_at": datetime.now(timezone.utc).isoformat()
    })

def compute_knowledge_graph(threshold: float) -> dict:
    """
    Build a knowledge graph from all stored chunks.
    Nodes = chunks, Edges = cosine similarity above threshold.
    Uses numpy for efficient pairwise computation.
    """
    total = collection.count()
    if total == 0:
        return {"nodes": [], "edges": []}

    # Fetch all chunks with embeddings and metadata
    all_data = collection.get(
        include=["embeddings", "metadatas", "documents"],
        limit=total
    )

    ids = all_data["ids"]
    embeddings = np.array(all_data["embeddings"])
    metadatas = all_data["metadatas"]
    documents = all_data["documents"]

    # Build node list
    nodes = []
    for i, (chunk_id, meta, doc) in enumerate(zip(ids, metadatas, documents)):
        nodes.append({
            "id": chunk_id,
            "label": f"{meta.get('title', 'Note')}[{meta.get('chunk_index', i)}]",
            "note_title": meta.get("title", "Unknown"),
            "note_id": meta.get("note_id", ""),
            "chunk_index": meta.get("chunk_index", i),
            "text_preview": (doc[:120] + "...") if len(doc) > 120 else doc,
        })

    # Compute pairwise cosine similarity using normalized dot products
    # Normalize embeddings (sentence-transformers outputs are already unit-normed, but normalize anyway)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    normed = embeddings / norms

    # Similarity matrix: (N x N) — only compute upper triangle
    edges = []
    n = len(ids)
    if n > 1:
        sim_matrix = np.dot(normed, normed.T)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i][j])
                if sim >= threshold:
                    edges.append({
                        "source": ids[i],
                        "target": ids[j],
                        "weight": round(sim, 4),
                    })

    return {"nodes": nodes, "edges": edges}

# ─── API Routes ───

@api_router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check — all local services."""
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
    Ingest a note: chunk → embed → store in ChromaDB + MongoDB.
    Invalidates graph cache on new ingestion.
    """
    content = request.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    note_id = str(uuid.uuid4())
    title = request.title.strip() or f"Note {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"

    # Chunk the text into semantic segments
    chunks = chunk_text(content)
    if not chunks:
        raise HTTPException(status_code=400, detail="Content too short to process")

    # Generate embeddings locally (no network)
    embeddings = generate_embeddings(chunks)

    # Store vectors in ChromaDB
    chunk_ids = [f"{note_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"note_id": note_id, "title": title, "chunk_index": i} for i in range(len(chunks))]
    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    # Store metadata in MongoDB
    note_doc = {
        "id": note_id,
        "title": title,
        "source_type": request.source_type,
        "chunk_count": len(chunks),
        "char_count": len(content),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.notes.insert_one(note_doc)

    # Invalidate graph cache (collection changed)
    await db.graph_cache.delete_many({})

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
    """Ingest a .txt or .md file — reads content and delegates to ingest_note."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ("txt", "md"):
        raise HTTPException(status_code=400, detail="Only .txt and .md files supported")

    content_bytes = await file.read()
    content = content_bytes.decode("utf-8", errors="ignore").strip()

    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    req = NoteIngestRequest(
        title=title or file.filename,
        content=content,
        source_type="file"
    )
    return await ingest_note(req)

@api_router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    RAG query pipeline:
    1. Embed the question locally
    2. Retrieve top-k chunks from ChromaDB by cosine similarity
    3. Generate answer via Ollama (or extractive fallback)
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

    # Embed the question (local, no network)
    q_embedding = generate_embeddings([question])[0]

    # Retrieve top-k relevant chunks
    top_k = min(request.top_k, total)
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
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
            "relevance": round(1 - dist, 3)  # cosine similarity = 1 - cosine distance
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

@api_router.get("/notes", response_model=PaginatedNotesResponse)
async def list_notes(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    limit: int = Query(20, ge=1, le=100, description="Items per page")
):
    """
    List ingested notes with pagination.
    Returns paginated response with total count and page metadata.
    """
    total = await db.notes.count_documents({})
    total_pages = max(1, (total + limit - 1) // limit)

    skip = (page - 1) * limit
    notes = await db.notes.find(
        {}, {"_id": 0}
    ).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)

    return PaginatedNotesResponse(
        notes=[NoteResponse(**n) for n in notes],
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages
    )

@api_router.delete("/notes/{note_id}")
async def delete_note(note_id: str):
    """Delete a note and its vectors from ChromaDB. Invalidates graph cache."""
    result = await db.notes.delete_one({"id": note_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Note not found")

    # Remove vectors from ChromaDB
    try:
        all_ids = collection.get(where={"note_id": note_id})["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
    except Exception:
        pass  # best-effort cleanup

    # Invalidate graph cache
    await db.graph_cache.delete_many({})

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

@api_router.get("/graph", response_model=GraphResponse)
async def get_knowledge_graph(
    threshold: float = Query(
        default=None,
        ge=0.0, le=1.0,
        description="Cosine similarity threshold for edges (0.0-1.0)"
    )
):
    """
    Build and return the knowledge graph.
    Nodes = note chunks, Edges = pairs with cosine similarity above threshold.
    Uses cached results when the collection hasn't changed.
    """
    # Use configurable default threshold
    if threshold is None:
        threshold = GRAPH_SIMILARITY_THRESHOLD

    total = collection.count()
    if total == 0:
        return GraphResponse(nodes=[], edges=[], threshold=threshold, cached=False)

    # Check cache first (avoids expensive pairwise computation)
    collection_hash = _compute_collection_hash()
    cached = await get_cached_graph(collection_hash, threshold)

    if cached:
        return GraphResponse(
            nodes=[GraphNode(**n) for n in cached["nodes"]],
            edges=[GraphEdge(**e) for e in cached["edges"]],
            threshold=threshold,
            cached=True
        )

    # Compute graph (expensive for large collections)
    graph_data = compute_knowledge_graph(threshold)

    # Cache the result in MongoDB
    await save_graph_cache(collection_hash, threshold, graph_data["nodes"], graph_data["edges"])

    return GraphResponse(
        nodes=[GraphNode(**n) for n in graph_data["nodes"]],
        edges=[GraphEdge(**e) for e in graph_data["edges"]],
        threshold=threshold,
        cached=False
    )

# ─── App Setup ───

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Privacy: no user content logged
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
