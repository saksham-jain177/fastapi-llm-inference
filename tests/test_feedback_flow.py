
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
import os

# Set env vars
os.environ["USE_MOCK"] = "true"
os.environ["API_KEY"] = "test-key"

pytestmark = pytest.mark.anyio

from app.main import app

class TestFeedbackFlow:
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
        
    async def test_feedback_writes_to_mongo(self, client):
        """Verify that feedback submission triggers a Mongo write."""
        
        with patch("app.rag.data_collector.get_data_collector") as mock_get_collector:
            mock_collector = MagicMock()
            
            # Mock Mongo collection
            mock_collection = MagicMock()
            mock_collection.insert_one = AsyncMock() # insert_one is async
            mock_collector.mongo_collection = mock_collection
            
            # Mock Redis for rate limit (return None = no previous request)
            mock_redis = MagicMock()
            mock_redis.get = AsyncMock(return_value=None)
            mock_redis.set = AsyncMock()
            mock_collector.redis_client = mock_redis
            
            # Ensure log_interaction calls mongo_collection.insert_one
            # We must use the REAL log_interaction method to test the flow?
            # Or trust that main calls log_interaction and log_interaction calls mongo?
            # If we mock log_interaction, we only test main -> log_interaction link.
            # If we want to test "Learning Loop", we should test main -> log_interaction -> mongo.
            # So we should use a real instance of DataCollector or mock only the DBs.
            
            # Real DataCollector with mocked DBs
            from app.rag.data_collector import DataCollector
            real_collector = DataCollector()
            real_collector.mongo_collection = mock_collection
            real_collector.redis_client = mock_redis
            
            mock_get_collector.return_value = real_collector
            
            payload = {
                "query": "What is the capital of Mars?",
                "response": "Elon Musk City",
                "label": "incorrect"
            }
            
            response = client.post("/feedback", json=payload)
            
            assert response.status_code == 200
            assert response.json()["status"] == "recorded"
            
            # Verify Mongo Write
            mock_collection.insert_one.assert_called_once()
            call_args = mock_collection.insert_one.call_args[0][0]
            assert call_args["query"] == "What is the capital of Mars?"
            assert call_args["feedback"] == "incorrect"
            
            # Verify Redis Rate Limit Set
            mock_redis.set.assert_called_once()
