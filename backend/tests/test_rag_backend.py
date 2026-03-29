"""Backend API tests for Privacy-First RAG Application

Tests cover:
- Health check endpoint
- Note ingestion (text)
- RAG query with retrieval
- Notes listing
- Note deletion
- System statistics
"""
import pytest
import requests
import time

class TestHealthEndpoint:
    """Health check and system status tests"""

    def test_health_check_returns_200(self, api_client, base_url):
        """GET /api/health should return 200 with system status"""
        response = api_client.get(f"{base_url}/api/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    def test_health_check_structure(self, api_client, base_url):
        """Health response should contain required fields"""
        response = api_client.get(f"{base_url}/api/health")
        data = response.json()
        
        assert "status" in data, "Missing 'status' field"
        assert "chromadb" in data, "Missing 'chromadb' field"
        assert "ollama" in data, "Missing 'ollama' field"
        assert "embedding_model" in data, "Missing 'embedding_model' field"
        assert "total_chunks" in data, "Missing 'total_chunks' field"
        
        assert data["status"] == "operational", f"Backend not operational: {data['status']}"
        assert data["chromadb"] == "connected", f"ChromaDB not connected: {data['chromadb']}"
        assert data["embedding_model"] == "all-MiniLM-L6-v2", f"Wrong embedding model: {data['embedding_model']}"
        assert isinstance(data["total_chunks"], int), "total_chunks should be integer"


class TestStatsEndpoint:
    """System statistics endpoint tests"""

    def test_stats_returns_200(self, api_client, base_url):
        """GET /api/stats should return 200"""
        response = api_client.get(f"{base_url}/api/stats")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    def test_stats_structure(self, api_client, base_url):
        """Stats response should contain required fields"""
        response = api_client.get(f"{base_url}/api/stats")
        data = response.json()
        
        assert "total_notes" in data, "Missing 'total_notes' field"
        assert "total_chunks" in data, "Missing 'total_chunks' field"
        assert "embedding_model" in data, "Missing 'embedding_model' field"
        assert "embedding_dim" in data, "Missing 'embedding_dim' field"
        assert "ollama_model" in data, "Missing 'ollama_model' field"
        
        assert isinstance(data["total_notes"], int), "total_notes should be integer"
        assert isinstance(data["total_chunks"], int), "total_chunks should be integer"
        assert data["embedding_dim"] == 384, f"Wrong embedding dimension: {data['embedding_dim']}"


class TestNoteIngestion:
    """Note ingestion and persistence tests"""

    def test_ingest_text_note_success(self, api_client, base_url):
        """POST /api/ingest should successfully ingest a text note"""
        payload = {
            "title": "TEST_Pytest_Note",
            "content": "This is a test note for pytest validation. It contains enough content to be chunked properly. Machine learning is a subset of artificial intelligence. Deep learning uses neural networks with multiple layers.",
            "source_type": "paste"
        }
        
        response = api_client.post(f"{base_url}/api/ingest", json=payload)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "id" in data, "Missing 'id' field in response"
        assert "title" in data, "Missing 'title' field in response"
        assert "chunk_count" in data, "Missing 'chunk_count' field in response"
        assert data["title"] == "TEST_Pytest_Note", f"Title mismatch: {data['title']}"
        assert data["chunk_count"] > 0, "Should have at least 1 chunk"

    def test_ingest_and_verify_persistence(self, api_client, base_url):
        """Create note and verify it persists in database via GET"""
        # Create note
        payload = {
            "title": "TEST_Persistence_Check",
            "content": "Testing data persistence in MongoDB and ChromaDB. This note should be retrievable after ingestion. Vector embeddings should be stored correctly.",
            "source_type": "paste"
        }
        
        create_response = api_client.post(f"{base_url}/api/ingest", json=payload)
        assert create_response.status_code == 200, f"Ingestion failed: {create_response.text}"
        
        created_note = create_response.json()
        note_id = created_note["id"]
        
        # Verify note appears in list
        time.sleep(0.5)  # Brief wait for DB write
        list_response = api_client.get(f"{base_url}/api/notes")
        assert list_response.status_code == 200, "Failed to fetch notes list"
        
        data = list_response.json()
        notes = data.get("notes", data) if isinstance(data, dict) else data
        note_ids = [n["id"] for n in notes]
        assert note_id in note_ids, f"Created note {note_id} not found in notes list"
        
        # Find the created note and verify fields
        created_note_from_list = next((n for n in notes if n["id"] == note_id), None)
        assert created_note_from_list is not None, "Note not found in list"
        assert created_note_from_list["title"] == "TEST_Persistence_Check", "Title mismatch"
        assert created_note_from_list["source_type"] == "paste", "Source type mismatch"
        assert created_note_from_list["chunk_count"] > 0, "Should have chunks"

    def test_ingest_empty_content_fails(self, api_client, base_url):
        """POST /api/ingest with empty content should return 400"""
        payload = {
            "title": "Empty Note",
            "content": "",
            "source_type": "paste"
        }
        
        response = api_client.post(f"{base_url}/api/ingest", json=payload)
        assert response.status_code == 400, f"Expected 400 for empty content, got {response.status_code}"

    def test_ingest_short_content_fails(self, api_client, base_url):
        """POST /api/ingest with very short content should return 400"""
        payload = {
            "title": "Too Short",
            "content": "Hi",
            "source_type": "paste"
        }
        
        response = api_client.post(f"{base_url}/api/ingest", json=payload)
        assert response.status_code == 400, f"Expected 400 for short content, got {response.status_code}"


class TestRAGQuery:
    """RAG query and retrieval tests"""

    def test_ask_question_returns_200(self, api_client, base_url):
        """POST /api/ask should return 200 with answer"""
        payload = {
            "question": "What is machine learning?",
            "top_k": 5
        }
        
        response = api_client.post(f"{base_url}/api/ask", json=payload)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    def test_ask_response_structure(self, api_client, base_url):
        """Ask response should contain answer, sources, and ollama_available"""
        payload = {
            "question": "Tell me about Python",
            "top_k": 3
        }
        
        response = api_client.post(f"{base_url}/api/ask", json=payload)
        data = response.json()
        
        assert "answer" in data, "Missing 'answer' field"
        assert "sources" in data, "Missing 'sources' field"
        assert "ollama_available" in data, "Missing 'ollama_available' field"
        
        assert isinstance(data["answer"], str), "Answer should be string"
        assert isinstance(data["sources"], list), "Sources should be list"
        assert isinstance(data["ollama_available"], bool), "ollama_available should be boolean"
        assert len(data["answer"]) > 0, "Answer should not be empty"

    def test_ask_with_existing_notes_returns_sources(self, api_client, base_url):
        """Query should return relevant sources from ingested notes"""
        # First ingest a note with known content
        ingest_payload = {
            "title": "TEST_RAG_Query_Note",
            "content": "Quantum computing uses qubits instead of classical bits. Qubits can exist in superposition states. Quantum entanglement enables quantum teleportation. Quantum algorithms like Shor's algorithm can factor large numbers efficiently.",
            "source_type": "paste"
        }
        
        ingest_response = api_client.post(f"{base_url}/api/ingest", json=ingest_payload)
        assert ingest_response.status_code == 200, "Failed to ingest test note"
        
        time.sleep(0.5)  # Wait for indexing
        
        # Query about the content
        ask_payload = {
            "question": "What is quantum computing?",
            "top_k": 5
        }
        
        response = api_client.post(f"{base_url}/api/ask", json=ask_payload)
        assert response.status_code == 200, "Query failed"
        
        data = response.json()
        assert len(data["sources"]) > 0, "Should return at least one source"
        
        # Verify source structure
        source = data["sources"][0]
        assert "text" in source, "Source missing 'text' field"
        assert "note_title" in source, "Source missing 'note_title' field"
        assert "relevance" in source, "Source missing 'relevance' field"
        assert isinstance(source["relevance"], (int, float)), "Relevance should be numeric"

    def test_ask_empty_question_fails(self, api_client, base_url):
        """POST /api/ask with empty question should return 400"""
        payload = {
            "question": "",
            "top_k": 5
        }
        
        response = api_client.post(f"{base_url}/api/ask", json=payload)
        assert response.status_code == 400, f"Expected 400 for empty question, got {response.status_code}"


class TestNotesManagement:
    """Notes listing and deletion tests"""

    def test_list_notes_returns_200(self, api_client, base_url):
        """GET /api/notes should return 200"""
        response = api_client.get(f"{base_url}/api/notes")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    def test_list_notes_returns_array(self, api_client, base_url):
        """GET /api/notes should return paginated response with notes array"""
        response = api_client.get(f"{base_url}/api/notes")
        data = response.json()
        
        # New paginated response format
        assert isinstance(data, dict), "Response should be dict with pagination"
        assert "notes" in data, "Response missing 'notes' field"
        assert isinstance(data["notes"], list), "notes should be array"
        
        if len(data["notes"]) > 0:
            note = data["notes"][0]
            assert "id" in note, "Note missing 'id' field"
            assert "title" in note, "Note missing 'title' field"
            assert "source_type" in note, "Note missing 'source_type' field"
            assert "chunk_count" in note, "Note missing 'chunk_count' field"
            assert "char_count" in note, "Note missing 'char_count' field"
            assert "created_at" in note, "Note missing 'created_at' field"

    def test_delete_note_success(self, api_client, base_url):
        """DELETE /api/notes/{id} should successfully delete note"""
        # Create a note to delete
        payload = {
            "title": "TEST_Delete_Me",
            "content": "This note will be deleted as part of the test. It contains sufficient content for chunking and embedding generation.",
            "source_type": "paste"
        }
        
        create_response = api_client.post(f"{base_url}/api/ingest", json=payload)
        assert create_response.status_code == 200, "Failed to create note for deletion test"
        
        note_id = create_response.json()["id"]
        time.sleep(0.5)
        
        # Delete the note
        delete_response = api_client.delete(f"{base_url}/api/notes/{note_id}")
        assert delete_response.status_code == 200, f"Delete failed: {delete_response.text}"
        
        delete_data = delete_response.json()
        assert "message" in delete_data, "Delete response missing 'message'"
        assert delete_data["id"] == note_id, "Deleted note ID mismatch"
        
        # Verify note is gone from list
        time.sleep(0.5)
        list_response = api_client.get(f"{base_url}/api/notes")
        data = list_response.json()
        notes = data.get("notes", data) if isinstance(data, dict) else data
        note_ids = [n["id"] for n in notes]
        assert note_id not in note_ids, f"Deleted note {note_id} still appears in list"

    def test_delete_nonexistent_note_fails(self, api_client, base_url):
        """DELETE /api/notes/{id} with invalid ID should return 404"""
        fake_id = "nonexistent-note-id-12345"
        response = api_client.delete(f"{base_url}/api/notes/{fake_id}")
        assert response.status_code == 404, f"Expected 404 for nonexistent note, got {response.status_code}"


class TestCleanup:
    """Cleanup test data created during tests"""

    def test_cleanup_test_notes(self, api_client, base_url):
        """Remove all notes created during testing (prefixed with TEST_)"""
        response = api_client.get(f"{base_url}/api/notes")
        if response.status_code != 200:
            pytest.skip("Cannot fetch notes for cleanup")
        
        data = response.json()
        notes = data.get("notes", data) if isinstance(data, dict) else data
        test_notes = [n for n in notes if n["title"].startswith("TEST_")]
        
        deleted_count = 0
        for note in test_notes:
            delete_response = api_client.delete(f"{base_url}/api/notes/{note['id']}")
            if delete_response.status_code == 200:
                deleted_count += 1
        
        print(f"\nCleaned up {deleted_count} test notes")
