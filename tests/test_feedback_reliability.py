
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
import os

# Set env vars before importing app
os.environ["USE_MOCK"] = "true"
os.environ["API_KEY"] = "test-key"

from app.main import app

class TestFeedbackReliability:
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
        
    def test_feedback_minimal_payload(self, client):
        """Test strict minimal payload (only required fields)."""
        with patch("app.rag.data_collector.get_data_collector") as mock_get_collector:
            mock_collector = MagicMock()
            mock_collector.redis_client = None # Bypass rate limit check
            mock_collector.log_interaction = AsyncMock()
            mock_get_collector.return_value = mock_collector
            
            payload = {
                "query": "Is this working?",
                "response": "Yes",
                "label": "correct"
            }
            
            response = client.post("/feedback", json=payload)
            assert response.status_code == 200
            assert response.json()["status"] == "recorded"

    def test_feedback_full_payload_with_nulls(self, client):
        """Test payload with optional fields set to null or present."""
        with patch("app.rag.data_collector.get_data_collector") as mock_get_collector:
            mock_collector = MagicMock()
            mock_collector.redis_client = None
            mock_collector.log_interaction = AsyncMock()
            mock_get_collector.return_value = mock_collector
            
            payload = {
                "query": "Is this working?",
                "response": "Yes",
                "label": "incorrect",
                "confidence": None,
                "model_mode": "rag",
                "intent": "external_search",
                "source": "rag"
            }
            
            response = client.post("/feedback", json=payload)
            assert response.status_code == 200
            
    def test_feedback_optional_missing_is_ok(self, client):
        """Test payload where optional fields are completely missing."""
        with patch("app.rag.data_collector.get_data_collector") as mock_get_collector:
            mock_collector = MagicMock()
            mock_collector.redis_client = None
            mock_collector.log_interaction = AsyncMock()
            mock_get_collector.return_value = mock_collector
            
            payload = {
                "query": "Q",
                "response": "A",
                "label": "correct"
            }
            # Should use defaults for model_mode etc.
            response = client.post("/feedback", json=payload)
            assert response.status_code == 200
