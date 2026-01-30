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
            
            assert result["source"] == "refused"
            assert "unavailable" in result["response"].lower()

    def test_infer_hard_limit(self, client):
        """Verify strict 413 limit on /infer endpoint."""
        oversized = "x" * 10000
        response = client.post("/infer", json={"prompt": oversized})
        assert response.status_code == 413
        assert "Too long" in response.json()["detail"] or "too long" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_adaptive_rate_limit(self):
        """Verify /infer-adaptive returns 429 on rate limit."""
        from app.main import check_rate_limit
        
        # Patch check_rate_limit to return False
        with patch("app.main.check_rate_limit", new_callable=AsyncMock) as mock_limit, \
             patch("app.routing.orchestrator.get_orchestrator") as mock_orch: # bypass orch
            
            mock_limit.return_value = False
            
            from fastapi.testclient import TestClient
            from app.main import app
            client = TestClient(app)
            
            response = client.post("/infer-adaptive", json={"prompt": "test"})
            
            assert response.status_code == 429
            assert "limit exceeded" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_rag_timeout_handling(self):
        """Verify orchestrator handles RAG timeout gracefully."""
        from app.routing.orchestrator import Orchestrator
        import asyncio
        
        # Patch wait_for to raise TimeoutError
        async def mock_wait_for(coro, timeout):
            raise asyncio.TimeoutError()
            
        with patch("app.routing.orchestrator.asyncio.wait_for", side_effect=mock_wait_for), \
             patch("app.routing.orchestrator.get_query_analyzer") as mock_analyzer, \
             patch("app.routing.orchestrator.search_web_context") as mock_search, \
             patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
             
             # Force intent
             mock_an = MagicMock()
             mock_an.analyze.return_value = {"intent": "external_search"}
             mock_analyzer.return_value = mock_an
             
             orch = Orchestrator()
             result = await orch.route_and_execute("Timeout query")
             
             assert result["refused"] is True
             assert "timed out" in result["response"]
