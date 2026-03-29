"""Backend API tests for NEW features: Knowledge Graph & Pagination

Tests cover:
- GET /api/graph - graph generation with caching
- GET /api/graph?threshold=X - adjustable threshold parameter
- Graph cache invalidation on ingest/delete
- GET /api/notes pagination - page, limit, total_pages
"""
import pytest
import requests
import time

class TestGraphEndpoint:
    """Knowledge graph endpoint tests"""

    def test_graph_returns_200(self, api_client, base_url):
        """GET /api/graph should return 200"""
        response = api_client.get(f"{base_url}/api/graph")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    def test_graph_response_structure(self, api_client, base_url):
        """Graph response should contain nodes, edges, threshold, cached fields"""
        response = api_client.get(f"{base_url}/api/graph")
        data = response.json()
        
        assert "nodes" in data, "Missing 'nodes' field"
        assert "edges" in data, "Missing 'edges' field"
        assert "threshold" in data, "Missing 'threshold' field"
        assert "cached" in data, "Missing 'cached' field"
        
        assert isinstance(data["nodes"], list), "nodes should be list"
        assert isinstance(data["edges"], list), "edges should be list"
        assert isinstance(data["threshold"], (int, float)), "threshold should be numeric"
        assert isinstance(data["cached"], bool), "cached should be boolean"

    def test_graph_with_custom_threshold(self, api_client, base_url):
        """GET /api/graph?threshold=0.3 should use custom threshold"""
        response = api_client.get(f"{base_url}/api/graph?threshold=0.3")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data["threshold"] == 0.3, f"Expected threshold 0.3, got {data['threshold']}"

    def test_graph_caching_works(self, api_client, base_url):
        """Second call to /api/graph with same threshold should return cached=true"""
        # First call - should compute
        response1 = api_client.get(f"{base_url}/api/graph?threshold=0.3")
        assert response1.status_code == 200, "First graph call failed"
        data1 = response1.json()
        
        # Second call - should be cached
        time.sleep(0.2)
        response2 = api_client.get(f"{base_url}/api/graph?threshold=0.3")
        assert response2.status_code == 200, "Second graph call failed"
        data2 = response2.json()
        
        assert data2["cached"] == True, f"Expected cached=true on second call, got {data2['cached']}"
        
        # Verify same data returned
        assert len(data1["nodes"]) == len(data2["nodes"]), "Node count mismatch between cached calls"
        assert len(data1["edges"]) == len(data2["edges"]), "Edge count mismatch between cached calls"

    def test_graph_node_structure(self, api_client, base_url):
        """Graph nodes should have required fields"""
        response = api_client.get(f"{base_url}/api/graph")
        data = response.json()
        
        if len(data["nodes"]) > 0:
            node = data["nodes"][0]
            assert "id" in node, "Node missing 'id' field"
            assert "label" in node, "Node missing 'label' field"
            assert "note_title" in node, "Node missing 'note_title' field"
            assert "note_id" in node, "Node missing 'note_id' field"
            assert "chunk_index" in node, "Node missing 'chunk_index' field"
            assert "text_preview" in node, "Node missing 'text_preview' field"

    def test_graph_edge_structure(self, api_client, base_url):
        """Graph edges should have source, target, weight"""
        response = api_client.get(f"{base_url}/api/graph")
        data = response.json()
        
        if len(data["edges"]) > 0:
            edge = data["edges"][0]
            assert "source" in edge, "Edge missing 'source' field"
            assert "target" in edge, "Edge missing 'target' field"
            assert "weight" in edge, "Edge missing 'weight' field"
            assert isinstance(edge["weight"], (int, float)), "Edge weight should be numeric"
            assert 0 <= edge["weight"] <= 1, f"Edge weight should be 0-1, got {edge['weight']}"

    def test_graph_cache_invalidation_on_ingest(self, api_client, base_url):
        """POST /api/ingest should invalidate graph cache"""
        # Get graph and verify it's cached
        response1 = api_client.get(f"{base_url}/api/graph?threshold=0.4")
        assert response1.status_code == 200, "Initial graph call failed"
        
        time.sleep(0.2)
        response2 = api_client.get(f"{base_url}/api/graph?threshold=0.4")
        data2 = response2.json()
        assert data2["cached"] == True, "Graph should be cached before ingest"
        
        # Ingest a new note
        ingest_payload = {
            "title": "TEST_Cache_Invalidation",
            "content": "This note should invalidate the graph cache. Machine learning models require training data. Neural networks use backpropagation for learning.",
            "source_type": "paste"
        }
        ingest_response = api_client.post(f"{base_url}/api/ingest", json=ingest_payload)
        assert ingest_response.status_code == 200, "Ingest failed"
        
        time.sleep(0.5)
        
        # Get graph again - should NOT be cached (cache invalidated)
        response3 = api_client.get(f"{base_url}/api/graph?threshold=0.4")
        data3 = response3.json()
        assert data3["cached"] == False, f"Graph cache should be invalidated after ingest, got cached={data3['cached']}"

    def test_graph_cache_invalidation_on_delete(self, api_client, base_url):
        """DELETE /api/notes/{id} should invalidate graph cache"""
        # Create a note to delete
        ingest_payload = {
            "title": "TEST_Delete_For_Cache",
            "content": "This note will be deleted to test cache invalidation. Deep learning uses convolutional neural networks for image processing.",
            "source_type": "paste"
        }
        ingest_response = api_client.post(f"{base_url}/api/ingest", json=ingest_payload)
        assert ingest_response.status_code == 200, "Failed to create note"
        note_id = ingest_response.json()["id"]
        
        time.sleep(0.5)
        
        # Get graph and cache it
        response1 = api_client.get(f"{base_url}/api/graph?threshold=0.5")
        assert response1.status_code == 200, "Initial graph call failed"
        
        time.sleep(0.2)
        response2 = api_client.get(f"{base_url}/api/graph?threshold=0.5")
        data2 = response2.json()
        assert data2["cached"] == True, "Graph should be cached before delete"
        
        # Delete the note
        delete_response = api_client.delete(f"{base_url}/api/notes/{note_id}")
        assert delete_response.status_code == 200, "Delete failed"
        
        time.sleep(0.5)
        
        # Get graph again - should NOT be cached
        response3 = api_client.get(f"{base_url}/api/graph?threshold=0.5")
        data3 = response3.json()
        assert data3["cached"] == False, f"Graph cache should be invalidated after delete, got cached={data3['cached']}"


class TestNotesPagination:
    """Pagination tests for GET /api/notes"""

    def test_notes_pagination_default(self, api_client, base_url):
        """GET /api/notes should return paginated response with default page=1, limit=20"""
        response = api_client.get(f"{base_url}/api/notes")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "notes" in data, "Missing 'notes' field"
        assert "total" in data, "Missing 'total' field"
        assert "page" in data, "Missing 'page' field"
        assert "limit" in data, "Missing 'limit' field"
        assert "total_pages" in data, "Missing 'total_pages' field"
        
        assert isinstance(data["notes"], list), "notes should be list"
        assert isinstance(data["total"], int), "total should be integer"
        assert data["page"] == 1, f"Default page should be 1, got {data['page']}"
        assert data["limit"] == 20, f"Default limit should be 20, got {data['limit']}"

    def test_notes_pagination_custom_limit(self, api_client, base_url):
        """GET /api/notes?page=1&limit=2 should return 2 notes per page"""
        response = api_client.get(f"{base_url}/api/notes?page=1&limit=2")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data["limit"] == 2, f"Expected limit=2, got {data['limit']}"
        assert len(data["notes"]) <= 2, f"Should return max 2 notes, got {len(data['notes'])}"

    def test_notes_pagination_second_page(self, api_client, base_url):
        """GET /api/notes?page=2&limit=2 should return different notes than page 1"""
        # Get page 1
        response1 = api_client.get(f"{base_url}/api/notes?page=1&limit=2")
        assert response1.status_code == 200, "Page 1 request failed"
        data1 = response1.json()
        
        # Get page 2
        response2 = api_client.get(f"{base_url}/api/notes?page=2&limit=2")
        assert response2.status_code == 200, "Page 2 request failed"
        data2 = response2.json()
        
        assert data2["page"] == 2, f"Expected page=2, got {data2['page']}"
        
        # If there are enough notes, verify different notes returned
        if data1["total"] > 2:
            page1_ids = [n["id"] for n in data1["notes"]]
            page2_ids = [n["id"] for n in data2["notes"]]
            
            # Pages should have different notes (no overlap)
            overlap = set(page1_ids) & set(page2_ids)
            assert len(overlap) == 0, f"Page 1 and Page 2 should have different notes, found overlap: {overlap}"

    def test_notes_pagination_total_pages_calculation(self, api_client, base_url):
        """total_pages should be correctly calculated"""
        response = api_client.get(f"{base_url}/api/notes?page=1&limit=3")
        assert response.status_code == 200, "Request failed"
        
        data = response.json()
        total = data["total"]
        limit = data["limit"]
        total_pages = data["total_pages"]
        
        # Calculate expected total_pages
        expected_pages = max(1, (total + limit - 1) // limit)
        assert total_pages == expected_pages, f"Expected {expected_pages} total pages, got {total_pages}"

    def test_notes_pagination_empty_page(self, api_client, base_url):
        """Requesting page beyond total_pages should return empty notes list"""
        response = api_client.get(f"{base_url}/api/notes?page=9999&limit=10")
        assert response.status_code == 200, "Request failed"
        
        data = response.json()
        assert len(data["notes"]) == 0, "Should return empty list for page beyond total_pages"
        assert data["page"] == 9999, "Should return requested page number"


class TestCleanup:
    """Cleanup test data created during graph/pagination tests"""

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
        
        print(f"\nCleaned up {deleted_count} test notes from graph/pagination tests")
