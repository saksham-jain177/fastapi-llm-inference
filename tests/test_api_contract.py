import pytest
import os
from unittest.mock import MagicMock, patch, AsyncMock
from app.main import app, InferenceRequest
from app.routing.orchestrator import Orchestrator

# Force deterministic modes
os.environ["USE_DETERMINISTIC_INFERENCE"] = "true"
os.environ["API_KEY"] = "test-secret"

@pytest.mark.asyncio
async def test_api_contract_response_shape():
    """
    Verify that the API returns the correct shape and types:
    - answer: str
    - confidence: float
    - intent: str
    - source: str
    - refused: bool
    """
    
    # Mock Orchestrator to return a known deterministic result
    # We patch where it is defined since it is imported locally in the endpoint
    with patch("app.routing.orchestrator.get_orchestrator") as mock_get_orch:
        mock_orch = MagicMock()
        mock_orch.route_and_execute = AsyncMock(return_value={
            "response": "Deterministic Answer",
            "confidence": 0.95,
            "intent": "simple_internal",
            "source": "model",
            "refused": False
        })
        mock_get_orch.return_value = mock_orch
        
        # We need to simulate the request context if we were calling the endpoint function directly,
        # but using TestClient is better for integration-like contract testing.
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # 1. Test standard successful response
        response = client.post(
            "/infer-adaptive", 
            json={"prompt": "test query"},
            headers={"Authorization": "Bearer test-secret"} # Main doesn't actually check header, checks env var.
        )
        
        # Verify 200 OK
        assert response.status_code == 200
        data = response.json()
        
        # Verify Keys
        assert "answer" in data
        assert "confidence" in data
        assert "intent" in data
        assert "source" in data
        assert "refused" in data
        
        # Verify Types/Values
        assert isinstance(data["confidence"], float)
        assert isinstance(data["refused"], bool)
        assert data["answer"] == "Deterministic Answer"
        assert data["source"] == "model"
        assert "citations" in data
        assert isinstance(data["citations"], list)
        
@pytest.mark.asyncio
async def test_api_contract_refusal():
    """Verify refusal flag behavior."""
    
    with patch("app.routing.orchestrator.get_orchestrator") as mock_get_orch:
        mock_orch = MagicMock()
        mock_orch.route_and_execute = AsyncMock(return_value={
            "response": "I cannot answer that.",
            "confidence": 0.4,
            "intent": "simple_internal",
            "source": "refused",
            "refused": True
        })
        mock_get_orch.return_value = mock_orch
        
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        response = client.post("/infer-adaptive", json={"prompt": "unsafe"})
        data = response.json()
        
        assert data["refused"] is True
        assert data["source"] == "refused"
        assert data["confidence"] == 0.4
