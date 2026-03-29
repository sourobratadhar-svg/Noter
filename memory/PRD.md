# Privacy-First RAG Notes App - PRD

## Overview
A fully local, privacy-first Retrieval-Augmented Generation (RAG) application that indexes and retrieves information from personal notes. All data processing happens on-device with no external API calls.

## Architecture
- **Frontend**: React Native (Expo) with Swiss brutalist design
- **Backend**: Python FastAPI
- **Vector Database**: ChromaDB (persistent local storage)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
- **LLM**: Ollama (local, with extractive fallback when unavailable)
- **Metadata Store**: MongoDB

## Features (MVP)

### 1. Notes Ingestion
- Paste text directly into the app
- Upload .txt/.md files from device storage
- Automatic chunking into semantically meaningful segments (300-500 tokens)
- Local embedding generation using sentence-transformers

### 2. Vector Indexing
- ChromaDB persistent collection with cosine similarity
- Embeddings generated locally (no network calls)
- Mapping maintained between chunks and original notes

### 3. RAG Query
- Semantic search via top-k chunk retrieval
- Grounded prompt construction (answers only from context)
- Ollama LLM integration for generated answers
- Extractive fallback when Ollama is unavailable

### 4. API Layer
- `POST /api/ingest` - Ingest text notes
- `POST /api/ingest-file` - Ingest .txt/.md files
- `POST /api/ask` - RAG query with answer + sources
- `GET /api/notes` - List ingested notes
- `DELETE /api/notes/{id}` - Delete note and vectors
- `GET /api/health` - System health check
- `GET /api/stats` - Collection statistics

### 5. Frontend (3 Tabs)
- **CHAT**: Ask questions, see RAG answers with source snippets
- **NOTES**: Add/view/delete notes
- **SYS**: Health status, stats, configuration, Ollama setup guide

## Privacy & Security
- All data stored locally
- No external API calls
- No telemetry or tracking
- No user content logging

## Tech Stack
- Expo SDK 54, React Native 0.81
- FastAPI, ChromaDB, sentence-transformers
- MongoDB for metadata
- Ollama for local LLM (optional)
