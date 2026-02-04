import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "mongo" in data
    assert "redis" in data

def test_infer_with_mock_env():
    import os
    from unittest.mock import patch
    from app.reasoners.factory import reset_reasoner
    from app.routing.semantic_router import reset_semantic_router
    from app.models.adapter_manager import reset_adapter_manager
    
    # Reset to ensure we don't use a cached non-deterministic components
    reset_reasoner()
    reset_semantic_router()
    reset_adapter_manager()
    
    # Mock the environment variable just for this test
    with patch.dict(os.environ, {"API_KEY": "test-secret", "USE_DETERMINISTIC_INFERENCE": "true"}):
        response = client.post("/infer", json={"prompt": "test"})
        assert response.status_code == 200
        assert "answer" in response.json()
