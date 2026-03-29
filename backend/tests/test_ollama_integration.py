"""Backend API tests for Enhanced Ollama Integration (Iteration 3)

Tests cover:
- Enhanced /api/health with Ollama fields
- New /api/ollama/status endpoint
- New POST /api/ollama/model endpoint
- Enhanced /api/ask with mode/model/ollama_error fields
"""
import pytest
import requests


class TestEnhancedHealthEndpoint:
    """Enhanced health check with Ollama diagnostics"""

    def test_health_returns_ollama_fields(self, api_client, base_url):
        """GET /api/health should return new Ollama diagnostic fields"""
        response = api_client.get(f"{base_url}/api/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        
        # New fields for iteration 3
        assert "ollama_error" in data, "Missing 'ollama_error' field"
        assert "ollama_models" in data, "Missing 'ollama_models' field"
        assert "ollama_active_model" in data, "Missing 'ollama_active_model' field"
        assert "ollama_model_loaded" in data, "Missing 'ollama_model_loaded' field"
        
        # Validate types
        assert isinstance(data["ollama_models"], list), "ollama_models should be list"
        assert isinstance(data["ollama_active_model"], str), "ollama_active_model should be string"
        assert isinstance(data["ollama_model_loaded"], bool), "ollama_model_loaded should be boolean"
        
        # In test env, Ollama is not running
        assert data["ollama"] == "unavailable", f"Expected Ollama unavailable, got {data['ollama']}"
        assert data["ollama_error"] is not None, "Should have ollama_error when unavailable"
        assert data["ollama_model_loaded"] is False, "Model should not be loaded when Ollama unavailable"

    def test_health_ollama_error_message(self, api_client, base_url):
        """Ollama error should contain helpful message"""
        response = api_client.get(f"{base_url}/api/health")
        data = response.json()
        
        error_msg = data.get("ollama_error", "")
        assert error_msg, "Should have error message when Ollama unavailable"
        assert len(error_msg) > 10, "Error message should be descriptive"
        # Should mention connection or running
        assert any(word in error_msg.lower() for word in ["connect", "running", "serve"]), \
            f"Error message should be helpful: {error_msg}"


class TestOllamaStatusEndpoint:
    """New /api/ollama/status endpoint tests"""

    def test_ollama_status_returns_200(self, api_client, base_url):
        """GET /api/ollama/status should return 200"""
        response = api_client.get(f"{base_url}/api/ollama/status")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    def test_ollama_status_structure(self, api_client, base_url):
        """Status response should contain all required fields"""
        response = api_client.get(f"{base_url}/api/ollama/status")
        data = response.json()
        
        # Required fields
        assert "available" in data, "Missing 'available' field"
        assert "models" in data, "Missing 'models' field"
        assert "active_model" in data, "Missing 'active_model' field"
        assert "model_loaded" in data, "Missing 'model_loaded' field"
        assert "error" in data, "Missing 'error' field"
        assert "base_url" in data, "Missing 'base_url' field"
        assert "troubleshooting" in data, "Missing 'troubleshooting' field"
        
        # Validate types
        assert isinstance(data["available"], bool), "available should be boolean"
        assert isinstance(data["models"], list), "models should be list"
        assert isinstance(data["active_model"], str), "active_model should be string"
        assert isinstance(data["model_loaded"], bool), "model_loaded should be boolean"
        assert isinstance(data["base_url"], str), "base_url should be string"
        assert isinstance(data["troubleshooting"], dict), "troubleshooting should be dict"

    def test_ollama_status_troubleshooting_hints(self, api_client, base_url):
        """Troubleshooting section should contain helpful hints"""
        response = api_client.get(f"{base_url}/api/ollama/status")
        data = response.json()
        
        troubleshooting = data["troubleshooting"]
        
        # Should have hints for common issues
        assert "not_running" in troubleshooting, "Missing 'not_running' hint"
        assert "no_model" in troubleshooting, "Missing 'no_model' hint"
        assert "network" in troubleshooting, "Missing 'network' hint"
        
        # Hints should be non-empty strings
        assert len(troubleshooting["not_running"]) > 5, "not_running hint too short"
        assert len(troubleshooting["no_model"]) > 5, "no_model hint too short"
        assert len(troubleshooting["network"]) > 5, "network hint too short"
        
        # Should mention ollama serve
        assert "ollama serve" in troubleshooting["not_running"], "Should mention 'ollama serve'"

    def test_ollama_status_when_unavailable(self, api_client, base_url):
        """Status should correctly report unavailable state"""
        response = api_client.get(f"{base_url}/api/ollama/status")
        data = response.json()
        
        # In test env, Ollama is not running
        assert data["available"] is False, "Ollama should be unavailable in test env"
        assert data["model_loaded"] is False, "Model should not be loaded"
        assert data["error"] is not None, "Should have error message"
        assert len(data["models"]) == 0, "Should have no models when unavailable"


class TestOllamaModelSwitching:
    """POST /api/ollama/model endpoint tests"""

    def test_set_model_returns_200(self, api_client, base_url):
        """POST /api/ollama/model should return 200"""
        payload = {"model": "llama3"}
        response = api_client.post(f"{base_url}/api/ollama/model", json=payload)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    def test_set_model_response_structure(self, api_client, base_url):
        """Model switch response should contain success, model, message"""
        payload = {"model": "mistral"}
        response = api_client.post(f"{base_url}/api/ollama/model", json=payload)
        data = response.json()
        
        assert "success" in data, "Missing 'success' field"
        assert "model" in data, "Missing 'model' field"
        assert "message" in data, "Missing 'message' field"
        
        assert isinstance(data["success"], bool), "success should be boolean"
        assert isinstance(data["model"], str), "model should be string"
        assert isinstance(data["message"], str), "message should be string"
        
        assert data["success"] is True, "Should succeed even when Ollama unavailable"
        assert data["model"] == "mistral", f"Model should be 'mistral', got {data['model']}"

    def test_set_model_persists(self, api_client, base_url):
        """Model change should persist and be reflected in health check"""
        # Set model to phi3
        payload = {"model": "phi3"}
        set_response = api_client.post(f"{base_url}/api/ollama/model", json=payload)
        assert set_response.status_code == 200, "Failed to set model"
        
        # Verify via health endpoint
        health_response = api_client.get(f"{base_url}/api/health")
        health_data = health_response.json()
        
        assert health_data["ollama_active_model"] == "phi3", \
            f"Active model should be 'phi3', got {health_data['ollama_active_model']}"

    def test_set_model_multiple_switches(self, api_client, base_url):
        """Should handle multiple model switches"""
        models = ["mistral", "llama3", "gemma2"]
        
        for model in models:
            payload = {"model": model}
            response = api_client.post(f"{base_url}/api/ollama/model", json=payload)
            assert response.status_code == 200, f"Failed to set model to {model}"
            
            data = response.json()
            assert data["model"] == model, f"Model mismatch: expected {model}, got {data['model']}"


class TestEnhancedAskEndpoint:
    """Enhanced /api/ask with mode, model, ollama_error fields"""

    def test_ask_returns_mode_field(self, api_client, base_url):
        """POST /api/ask should return 'mode' field"""
        payload = {"question": "What is machine learning?", "top_k": 5}
        response = api_client.post(f"{base_url}/api/ask", json=payload)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "mode" in data, "Missing 'mode' field"
        assert isinstance(data["mode"], str), "mode should be string"
        
        # In test env with Ollama unavailable, should be extractive
        assert data["mode"] in ["extractive", "ollama", "none"], \
            f"Invalid mode: {data['mode']}"

    def test_ask_returns_model_field(self, api_client, base_url):
        """POST /api/ask should return 'model' field"""
        payload = {"question": "Tell me about Python", "top_k": 3}
        response = api_client.post(f"{base_url}/api/ask", json=payload)
        data = response.json()
        
        assert "model" in data, "Missing 'model' field"
        assert isinstance(data["model"], str), "model should be string"
        # Should return the active model name even if Ollama unavailable
        assert len(data["model"]) > 0, "model field should not be empty"

    def test_ask_returns_ollama_error_when_unavailable(self, api_client, base_url):
        """POST /api/ask should return ollama_error when Ollama unavailable"""
        payload = {"question": "What is quantum computing?", "top_k": 5}
        response = api_client.post(f"{base_url}/api/ask", json=payload)
        data = response.json()
        
        assert "ollama_error" in data, "Missing 'ollama_error' field"
        
        # In test env, Ollama is unavailable
        if data["ollama_available"] is False:
            assert data["ollama_error"] is not None, "Should have error when unavailable"
            assert len(data["ollama_error"]) > 5, "Error message should be descriptive"

    def test_ask_extractive_mode_when_ollama_down(self, api_client, base_url):
        """When Ollama unavailable, should use extractive mode"""
        # First ingest a note
        ingest_payload = {
            "title": "TEST_Ollama_Mode_Check",
            "content": "Artificial intelligence is the simulation of human intelligence by machines. Machine learning is a subset of AI that enables systems to learn from data.",
            "source_type": "paste"
        }
        ingest_response = api_client.post(f"{base_url}/api/ingest", json=ingest_payload)
        assert ingest_response.status_code == 200, "Failed to ingest test note"
        
        # Query
        ask_payload = {"question": "What is AI?", "top_k": 5}
        response = api_client.post(f"{base_url}/api/ask", json=ask_payload)
        data = response.json()
        
        # Should use extractive mode when Ollama unavailable
        assert data["ollama_available"] is False, "Ollama should be unavailable in test env"
        assert data["mode"] == "extractive", f"Expected extractive mode, got {data['mode']}"
        assert data["answer"], "Should still return answer in extractive mode"
        assert len(data["sources"]) > 0, "Should return sources in extractive mode"


class TestBackwardCompatibility:
    """Ensure existing endpoints still work after refactor"""

    def test_ingest_still_works(self, api_client, base_url):
        """POST /api/ingest should still work with modular architecture"""
        payload = {
            "title": "TEST_Modular_Ingest",
            "content": "Testing that chunking.py, embeddings.py, and rag.py modules work correctly. This content should be chunked and embedded properly.",
            "source_type": "paste"
        }
        response = api_client.post(f"{base_url}/api/ingest", json=payload)
        assert response.status_code == 200, f"Ingest failed: {response.text}"
        
        data = response.json()
        assert data["chunk_count"] > 0, "Should have chunks"

    def test_notes_pagination_still_works(self, api_client, base_url):
        """GET /api/notes should still return paginated response"""
        response = api_client.get(f"{base_url}/api/notes?page=1&limit=20")
        assert response.status_code == 200, "Notes endpoint failed"
        
        data = response.json()
        assert "notes" in data, "Missing notes field"
        assert "total" in data, "Missing total field"
        assert "page" in data, "Missing page field"

    def test_graph_still_works(self, api_client, base_url):
        """GET /api/graph should still work"""
        response = api_client.get(f"{base_url}/api/graph")
        assert response.status_code == 200, "Graph endpoint failed"
        
        data = response.json()
        assert "nodes" in data, "Missing nodes field"
        assert "edges" in data, "Missing edges field"

    def test_stats_still_works(self, api_client, base_url):
        """GET /api/stats should still work"""
        response = api_client.get(f"{base_url}/api/stats")
        assert response.status_code == 200, "Stats endpoint failed"
        
        data = response.json()
        assert "total_notes" in data, "Missing total_notes"
        assert "ollama_model" in data, "Missing ollama_model"


class TestCleanup:
    """Cleanup test data"""

    def test_cleanup_test_notes(self, api_client, base_url):
        """Remove all notes created during testing"""
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
        
        print(f"\nCleaned up {deleted_count} test notes from Ollama integration tests")
