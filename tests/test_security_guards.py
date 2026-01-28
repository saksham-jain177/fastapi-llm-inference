"""
Tests for new security guards: prompt size limits, rate limiting, and RAG capability.
All tests use deterministic mode and existing mocking patterns.
"""

import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Set required env vars BEFORE importing app
os.environ["API_KEY"] = "test-key"
os.environ["USE_MOCK"] = "true"

from app.main import app


class TestSecurityGuards:
    """Test suite for hardened inference endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_oversized_prompt_refused(self, client):
        """Verify oversized prompts are rejected with refusal response."""
        # Generate prompt exceeding MAX_PROMPT_LENGTH (8192 default)
        oversized_prompt = "x" * 10000
        
        response = client.post("/infer-adaptive", json={"prompt": oversized_prompt})
        
        assert response.status_code == 200  # Returns refusal, not error
        data = response.json()
        assert data["refused"] is True
        assert data["source"] == "refused"
        assert "exceeds" in data["answer"].lower() or "length" in data["answer"].lower()
    
    @pytest.mark.asyncio
    async def test_rag_refused_without_tavily_key(self):
        """Verify RAG path refuses cleanly when TAVILY_API_KEY is missing."""
        from app.routing.orchestrator import Orchestrator
        
        # Ensure TAVILY_API_KEY is not set
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False), \
             patch("app.routing.orchestrator.get_query_analyzer") as mock_analyzer, \
             patch("app.routing.orchestrator.get_reasoner") as mock_reasoner:
            
            # Force external_search intent
            mock_analyzer_instance = MagicMock()
            mock_analyzer_instance.analyze.return_value = {"intent": "external_search"}
            mock_analyzer.return_value = mock_analyzer_instance
            mock_reasoner.return_value = MagicMock()
            
            orch = Orchestrator()
            result = await orch.route_and_execute("What is the latest news?")
            
            assert result["refused"] is True
            assert result["source"] == "refused"
            assert "unavailable" in result["response"].lower()
