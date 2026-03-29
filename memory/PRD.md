# Privacy-First RAG Notes App — PRD

## Overview
Mobile-first, fully local RAG application. Mobile app = interface, local backend = intelligence. All user data stays private.

## Architecture
```
Mobile (Expo) ←→ FastAPI Backend ←→ {ChromaDB, MongoDB, Ollama}
              (same WiFi network)        (all on laptop)
```

## Backend Modules
- **`llm.py`** — OllamaClient class: health checks, model listing, generation, model switching
- **`embeddings.py`** — EmbeddingEngine: sentence-transformers local encoding
- **`chunking.py`** — Semantic text splitting (paragraph → sentence → hard split)
- **`rag.py`** — RAGPipeline: embed → retrieve → prompt → generate (with extractive fallback)
- **`server.py`** — FastAPI routes, MongoDB, ChromaDB, graph computation

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Detailed health (backend, ChromaDB, Ollama + models) |
| `GET` | `/api/stats` | Collection statistics |
| `GET` | `/api/ollama/status` | Detailed Ollama diagnostics + troubleshooting |
| `POST` | `/api/ollama/model` | Switch active Ollama model at runtime |
| `POST` | `/api/ingest` | Ingest text note |
| `POST` | `/api/ingest-file` | Ingest .txt/.md file |
| `POST` | `/api/ask` | RAG query (answer + sources + mode + model) |
| `GET` | `/api/notes?page=1&limit=20` | Paginated notes list |
| `DELETE` | `/api/notes/{id}` | Delete note + vectors |
| `GET` | `/api/graph?threshold=0.65` | Knowledge graph |

## Frontend (4 Tabs)
- **CHAT**: RAG query with mode indicator (ollama/extractive), source snippets
- **NOTES**: Add/view/delete notes, infinite scroll pagination
- **GRAPH**: Interactive d3-force knowledge graph
- **SYS**: Health dashboard, Ollama diagnostics, model switcher, network setup guide

## Tech Stack
- Expo SDK 54, FastAPI, ChromaDB, sentence-transformers, Ollama, MongoDB, d3-force, numpy
