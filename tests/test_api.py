"""
Integration tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import json

from app.api import app


class TestAPI:
    """Test FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "app_name" in data
        assert "version" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "endpoints" in data
    
    def test_stats_endpoint(self, client):
        """Test stats endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "collection_name" in data
        assert "document_count" in data
    
    def test_ingest_unsupported_format(self, client):
        """Test ingestion with unsupported file format."""
        # Create a temporary .txt file (unsupported)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = Path(f.name)
        
        try:
            with open(temp_path, 'rb') as f:
                response = client.post(
                    "/ingest",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            
            # Should return 400 for unsupported format
            assert response.status_code == 400
        finally:
            temp_path.unlink()
    
    def test_query_endpoint(self, client):
        """Test query endpoint."""
        query_data = {
            "query": "What is the purpose of this system?",
            "top_k": 3
        }
        
        response = client.post("/query", json=query_data)
        
        # Should return 200 even if no documents are ingested
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "context_used" in data
        assert "retrieval_latency" in data
        assert "generation_latency" in data
        assert "total_latency" in data
    
    def test_query_with_filters(self, client):
        """Test query with metadata filters."""
        query_data = {
            "query": "test query",
            "category": "tech",
            "file_type": ".pdf",
            "top_k": 5
        }
        
        response = client.post("/query", json=query_data)
        assert response.status_code == 200
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "query_latency" in data
        assert "retrieval_latency" in data
        assert "generation_latency" in data
