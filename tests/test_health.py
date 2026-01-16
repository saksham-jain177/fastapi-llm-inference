import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_infer_with_mock_env():
    import os
    from unittest.mock import patch
    
    # Mock the environment variable just for this test
    with patch.dict(os.environ, {"API_KEY": "test-secret"}):
        response = client.post("/infer", json={"prompt": "test"})
        assert response.status_code == 200
        assert "response" in response.json()
