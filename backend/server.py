"""
Privacy-First RAG Notes Server
===============================
All processing happens locally. No external API calls.

Modular Architecture:
  - chunking.py  → semantic text splitting
  - embeddings.py → sentence-transformers local encoding
  - llm.py       → Ollama wrapper with health checks
  - rag.py       → full RAG pipeline (embed → retrieve → prompt → generate)
  - server.py    → FastAPI routes, MongoDB, ChromaDB, graph

Data flow:
  Mobile App → FastAPI → {ChromaDB, MongoDB, Ollama} → Mobile App
  All on localhost. No data leaves the machine.
"""

from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import chromadb
import numpy as np
import hashlib

# ─── Local Modules ───
from chunking import chunk_text
from embeddings import EmbeddingEngine
from llm import OllamaClient
from qwen_llm import QwenClient
from llm_router import LLMRouter
from rag import RAGPipeline

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

# ─── Module Initialization ───

# Embeddings: local sentence-transformers model
embedding_engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")

# LLM: Ollama client (talks to local Ollama instance)
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'mistral')
ollama_client = OllamaClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)

# Qwen LLM Client (Optional/parallel)
qwen_client = QwenClient()

# LLM Router handles dynamically switching engines
llm_router = LLMRouter(ollama_client, qwen_client)

# RAG: full pipeline combining embeddings + retrieval + LLM
rag_pipeline = RAGPipeline(embedding_engine, collection, llm_router)

# Graph config
GRAPH_SIMILARITY_THRESHOLD = float(os.environ.get('GRAPH_SIMILARITY_THRESHOLD', '0.65'))

app = FastAPI()
api_router = APIRouter(prefix="/api")

# ─── Pydantic Models ───

class NoteIngestRequest(BaseModel):
    title: str = ""
    content: str
    source_type: str = "paste"

class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    chat_history: List[dict] = []

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
    ollama_error: Optional[str] = None
    mode: str = "extractive"
    model: str = ""

class HealthResponse(BaseModel):
    status: str
    chromadb: str
    ollama: str
    ollama_error: Optional[str] = None
    ollama_models: List[str] = []
    ollama_active_model: str = ""
    ollama_model_loaded: bool = False
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

class OllamaModelRequest(BaseModel):
    model: str

class EngineSwitchRequest(BaseModel):
    engine: str

# ─── Knowledge Graph Module ───

def _compute_collection_hash() -> str:
    """Fast fingerprint of ChromaDB state for cache invalidation."""
    total = collection.count()
    if total == 0:
        return "empty"
    all_data = collection.get(limit=1, include=[])
    first_id = all_data["ids"][0] if all_data["ids"] else ""
    return hashlib.md5(f"{total}:{first_id}".encode()).hexdigest()

async def get_cached_graph(collection_hash: str, threshold: float):
    cache = await db.graph_cache.find_one(
        {"collection_hash": collection_hash, "threshold": threshold},
        {"_id": 0}
    )
    return cache

async def save_graph_cache(collection_hash: str, threshold: float, nodes: list, edges: list):
    await db.graph_cache.delete_many({"threshold": threshold})
    await db.graph_cache.insert_one({
        "collection_hash": collection_hash,
        "threshold": threshold,
        "nodes": nodes,
        "edges": edges,
        "computed_at": datetime.now(timezone.utc).isoformat()
    })

def compute_knowledge_graph(threshold: float) -> dict:
    """Build knowledge graph: nodes=chunks, edges=similarity above threshold."""
    total = collection.count()
    if total == 0:
        return {"nodes": [], "edges": []}

    all_data = collection.get(
        include=["embeddings", "metadatas", "documents"],
        limit=total
    )

    ids = all_data["ids"]
    embeddings = np.array(all_data["embeddings"])
    metadatas = all_data["metadatas"]
    documents = all_data["documents"]

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

    # Pairwise cosine similarity via normalized dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = embeddings / norms

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
    """
    Detailed system health check.
    Returns status of all local services: backend, ChromaDB, Ollama.
    Includes Ollama/Qwen model list and error details for troubleshooting.
    """
    ollama_status = await llm_router.check_health()
    total_chunks = collection.count()
    return HealthResponse(
        status="operational",
        chromadb="connected",
        ollama="connected" if ollama_status["available"] else "unavailable",
        ollama_error=ollama_status["error"],
        ollama_models=ollama_status["models"],
        ollama_active_model=ollama_status["active_model"],
        ollama_model_loaded=ollama_status["model_loaded"],
        embedding_model=embedding_engine.model_name,
        total_chunks=total_chunks
    )

@api_router.get("/ollama/status")
async def ollama_status():
    """
    Detailed Ollama status — used by the mobile app's system screen.
    Returns connection status, available models, and troubleshooting hints.
    """
    status = await ollama_client.check_health()
    return {
        "available": status["available"],
        "models": status["models"],
        "active_model": status["active_model"],
        "model_loaded": status["model_loaded"],
        "error": status["error"],
        "base_url": OLLAMA_BASE_URL,
        "troubleshooting": {
            "not_running": "Start Ollama with: ollama serve",
            "no_model": f"Pull a model with: ollama pull {ollama_client.model}",
            "network": "Ensure phone and laptop are on the same WiFi network",
        }
    }

@api_router.post("/engine")
async def switch_engine(req: EngineSwitchRequest):
    """
    Switch the active LLM engine between 'ollama' and 'qwen'.
    """
    if llm_router.set_engine(req.engine):
        return {"success": True, "engine": req.engine}
    raise HTTPException(status_code=400, detail="Invalid engine. Use 'ollama' or 'qwen'.")

@api_router.post("/ollama/model")
async def set_ollama_model(request: OllamaModelRequest):
    """
    Change the active Ollama model at runtime.
    The model must already be pulled in Ollama.
    """
    old_model = ollama_client.model
    ollama_client.set_model(request.model)

    # Verify the new model is available
    status = await ollama_client.check_health()
    if status["available"] and status["model_loaded"]:
        return {
            "success": True,
            "model": request.model,
            "message": f"Switched from {old_model} to {request.model}"
        }
    elif status["available"]:
        return {
            "success": True,
            "model": request.model,
            "message": f"Model set to {request.model} (not yet pulled — run: ollama pull {request.model})",
            "warning": "Model not found in Ollama"
        }
    else:
        # Still set it — Ollama might start later
        return {
            "success": True,
            "model": request.model,
            "message": f"Model set to {request.model} (Ollama not running)"
        }

@api_router.post("/ingest")
async def ingest_note(request: NoteIngestRequest):
    """
    Ingest a note: chunk → embed → store in ChromaDB + MongoDB.
    Pipeline: text → chunking.py → embeddings.py → ChromaDB + MongoDB
    """
    content = request.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    note_id = str(uuid.uuid4())
    title = request.title.strip() or f"Note {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"

    # Step 1: Chunk via chunking module
    chunks = chunk_text(content)
    if not chunks:
        raise HTTPException(status_code=400, detail="Content too short to process")

    # Step 2: Embed via embeddings module (local, no network)
    embeddings = embedding_engine.encode(chunks)

    # Step 3: Store vectors in ChromaDB
    chunk_ids = [f"{note_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"note_id": note_id, "title": title, "chunk_index": i} for i in range(len(chunks))]
    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    # Step 4: Store metadata in MongoDB
    note_doc = {
        "id": note_id,
        "title": title,
        "source_type": request.source_type,
        "chunk_count": len(chunks),
        "char_count": len(content),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.notes.insert_one(note_doc)

    # Invalidate graph cache
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

    req = NoteIngestRequest(title=title or file.filename, content=content, source_type="file")
    return await ingest_note(req)

@api_router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    RAG query endpoint.
    Delegates to RAGPipeline which handles:
      embed question → retrieve chunks → build prompt → call Ollama → fallback
    Returns answer, sources, and detailed Ollama status.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Delegate to RAG pipeline module
    result = await rag_pipeline.query(question, top_k=request.top_k, chat_history=request.chat_history)

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        ollama_available=result["ollama_available"],
        ollama_error=result.get("ollama_error"),
        mode=result.get("mode", "extractive"),
        model=result.get("model", ollama_client.model),
    )

@api_router.get("/notes", response_model=PaginatedNotesResponse)
async def list_notes(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    limit: int = Query(20, ge=1, le=100, description="Items per page")
):
    """Paginated notes list."""
    total = await db.notes.count_documents({})
    total_pages = max(1, (total + limit - 1) // limit)
    skip = (page - 1) * limit
    notes = await db.notes.find({}, {"_id": 0}).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)

    return PaginatedNotesResponse(
        notes=[NoteResponse(**n) for n in notes],
        total=total, page=page, limit=limit, total_pages=total_pages
    )

@api_router.delete("/notes/{note_id}")
async def delete_note(note_id: str):
    """Delete note + vectors. Invalidates graph cache."""
    result = await db.notes.delete_one({"id": note_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Note not found")

    try:
        all_ids = collection.get(where={"note_id": note_id})["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
    except Exception:
        pass

    await db.graph_cache.delete_many({})
    return {"message": "Note deleted", "id": note_id}

@api_router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Collection statistics."""
    note_count = await db.notes.count_documents({})
    chunk_count = collection.count()
    return StatsResponse(
        total_notes=note_count,
        total_chunks=chunk_count,
        embedding_model=embedding_engine.model_name,
        embedding_dim=embedding_engine.dimension,
        ollama_model=ollama_client.model
    )

@api_router.get("/graph", response_model=GraphResponse)
async def get_knowledge_graph(
    threshold: float = Query(default=None, ge=0.0, le=1.0)
):
    """Knowledge graph with cached edges."""
    if threshold is None:
        threshold = GRAPH_SIMILARITY_THRESHOLD

    total = collection.count()
    if total == 0:
        return GraphResponse(nodes=[], edges=[], threshold=threshold, cached=False)

    collection_hash = _compute_collection_hash()
    cached = await get_cached_graph(collection_hash, threshold)

    if cached:
        return GraphResponse(
            nodes=[GraphNode(**n) for n in cached["nodes"]],
            edges=[GraphEdge(**e) for e in cached["edges"]],
            threshold=threshold, cached=True
        )

    graph_data = compute_knowledge_graph(threshold)
    await save_graph_cache(collection_hash, threshold, graph_data["nodes"], graph_data["edges"])

    return GraphResponse(
        nodes=[GraphNode(**n) for n in graph_data["nodes"]],
        edges=[GraphEdge(**e) for e in graph_data["edges"]],
        threshold=threshold, cached=False
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
