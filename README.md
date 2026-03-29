# Noter — Privacy-First RAG Notes App

A fully local, privacy-first Retrieval-Augmented Generation (RAG) application that indexes and retrieves information from your personal notes. **All processing happens on-device** — no external API calls, no telemetry, no cloud services.

## Features

### RAG Chat
Ask natural language questions about your notes. The system retrieves relevant chunks via semantic search and generates grounded answers.

### Notes Ingestion
- Paste text directly into the app
- Upload `.txt` and `.md` files
- Automatic chunking into semantically meaningful segments (300-500 tokens)
- Local embedding generation using sentence-transformers

### Knowledge Graph
Interactive force-directed graph visualization showing relationships between note chunks. Nodes represent chunks, edges represent cosine similarity above a configurable threshold.

- Zoom, pan, and drag nodes
- Hover/tap to highlight connected nodes
- Adjustable similarity threshold
- Cached computation for fast rendering

### System Health Dashboard
Monitor ChromaDB, Ollama, and embedding model status. View statistics and configuration.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Native (Expo)                   │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐               │
│  │ CHAT │  │NOTES │  │GRAPH │  │ SYS  │               │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘               │
└─────┼─────────┼─────────┼─────────┼─────────────────────┘
      │         │         │         │
      ▼         ▼         ▼         ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend                         │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐             │
│  │Ingestion │  │ Retrieval │  │  Graph   │             │
│  │ Module   │  │  (RAG)    │  │ Module   │             │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘             │
│       │              │              │                    │
│  ┌────▼──────────────▼──────────────▼────┐              │
│  │        Embeddings Module              │              │
│  │   sentence-transformers (local)       │              │
│  └───────────────┬───────────────────────┘              │
│                  │                                       │
│  ┌───────────────▼───────────────────────┐              │
│  │         ChromaDB (persistent)          │              │
│  └───────────────────────────────────────┘              │
│                                                          │
│  ┌───────────────────────────────────────┐              │
│  │      Ollama (local LLM, optional)      │              │
│  └───────────────────────────────────────┘              │
│                                                          │
│  ┌───────────────────────────────────────┐              │
│  │         MongoDB (metadata)             │              │
│  └───────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | System health check |
| `GET` | `/api/stats` | Collection statistics |
| `POST` | `/api/ingest` | Ingest text note |
| `POST` | `/api/ingest-file` | Ingest .txt/.md file |
| `POST` | `/api/ask` | RAG query with answer + sources |
| `GET` | `/api/notes?page=1&limit=20` | Paginated notes list |
| `DELETE` | `/api/notes/{id}` | Delete note + vectors |
| `GET` | `/api/graph?threshold=0.65` | Knowledge graph (nodes + edges) |

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB (local instance)
- Ollama (optional, for LLM-powered answers)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your MongoDB URL

# Start the server
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

The embedding model (`all-MiniLM-L6-v2`) downloads automatically on first run (~80MB).

### Frontend Setup

```bash
cd frontend

# Install dependencies
yarn install

# Start Expo dev server
npx expo start
```

### Ollama Setup (Optional)

Ollama enables AI-generated answers instead of extractive fallback.

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull mistral

# Ollama runs automatically on localhost:11434
# The app auto-detects it
```

---

## Environment Variables

### Backend (`backend/.env`)
```
MONGO_URL=mongodb://localhost:27017
DB_NAME=rag_notes
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
GRAPH_SIMILARITY_THRESHOLD=0.65
```

### Frontend (`frontend/.env`)
```
EXPO_PUBLIC_BACKEND_URL=http://localhost:8001
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | React Native (Expo SDK 54) |
| Backend | Python FastAPI |
| Vector DB | ChromaDB (persistent local) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Ollama (mistral/llama3) |
| Metadata | MongoDB |
| Graph Viz | d3-force (via WebView) |

---

## Privacy & Security

- **All data stored locally** — ChromaDB on disk, MongoDB local instance
- **No external API calls** — embeddings generated on-device
- **No telemetry or tracking** — zero analytics
- **No user content logging** — privacy-first logging config
- **Ollama runs locally** — LLM inference without network access

---

## Graph View Screenshots

*The knowledge graph visualizes semantic relationships between your notes:*

| Empty State | Graph with Connections |
|---|---|
| ![Graph Empty](screenshots/graph-empty.png) | ![Graph Connected](screenshots/graph-connected.png) |

---

## Project Structure

```
noter/
├── backend/
│   ├── server.py          # FastAPI app with all modules
│   ├── .env               # Backend configuration
│   ├── requirements.txt   # Python dependencies
│   └── chroma_data/       # Persistent vector storage
├── frontend/
│   ├── app/
│   │   ├── _layout.tsx    # Tab navigation layout
│   │   ├── index.tsx      # Chat screen (RAG query)
│   │   ├── notes.tsx      # Notes management + ingestion
│   │   ├── graph.tsx      # Knowledge graph visualization
│   │   └── system.tsx     # System health dashboard
│   ├── .env               # Frontend configuration
│   ├── package.json       # Node dependencies
│   └── app.json           # Expo configuration
└── README.md
```

---

## Extensibility

The codebase is designed for modular extension:

- **Ingestion plugins**: Add new source types (Apple Notes, Notion) by implementing new ingestion endpoints
- **Embedding models**: Swap `all-MiniLM-L6-v2` for any sentence-transformers model
- **LLM backends**: Replace Ollama with any local LLM that exposes an HTTP API
- **Encryption**: Add AES encryption layer for ChromaDB storage
- **Authentication**: Add PIN/biometric auth as a middleware layer

---

## License

MIT
