"""
Tests for new security guards: prompt size limits, rate limiting, and RAG capability.
All tests use deterministic mode and existing mocking patterns.
"""

import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

pytestmark = pytest.mark.anyio

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
        # Use a high-information string that is also oversized
        oversized_prompt = "What are the specific technical architectural requirements for indexing a multi-terabyte database with high availability? " * 100
        
        response = client.post("/infer-adaptive", json={"prompt": oversized_prompt})
        
        assert response.status_code == 200  # Returns refusal, not error
        data = response.json()
        assert data["refused"] is True
        assert data["source"] == "refused"
        assert "exceeds" in data["answer"].lower() or "length" in data["answer"].lower()
    
    async def test_rag_fallback_without_tavily_key(self):
        """Verify RAG path attempts fallback when TAVILY_API_KEY is missing."""
        from app.routing.orchestrator import Orchestrator
        
        # Ensure TAVILY_API_KEY is not set
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False), \
             patch("app.routing.orchestrator.get_semantic_router") as mock_router, \
             patch("app.routing.orchestrator.get_reasoner") as mock_reasoner, \
             patch("app.routing.orchestrator.search_web_context") as mock_search, \
             patch("app.routing.orchestrator.get_context_packer") as mock_packer:
            
            # Force unknown domain to trigger RAG path
            mock_router_instance = MagicMock()
            mock_router_instance.classify.return_value = ("unknown", 0.2)
            mock_router.return_value = mock_router_instance
            
            # Search returns success (simulating fallback success happening inside retrieval)
            # note: in real run retrieval.py does the fallback logic, here we mock the result of search_web_context
            mock_search.return_value = ("Fallback successful.", [{"title": "DDG", "content": "Fallback Content", "url": "ddg.com"}])
            
            # Mock packer
            mock_packer_instance = MagicMock()
            mock_packer_instance.pack.return_value = ("Packed context", [{"title": "DDG", "content": "Fallback Content"}])
            mock_packer.return_value = mock_packer_instance
            
            # Mock reasoner
            mock_reasoner_instance = AsyncMock()
            mock_reasoner_instance.synthesize_with_context.return_value = "Answer from fallback."
            mock_reasoner.return_value = mock_reasoner_instance
            
            orch = Orchestrator()
            result = await orch.route_and_execute("What is the latest news?")
            
            # Should NOT be refused
            assert result["mode"] == "rag-external"
            assert result["response"] == "Answer from fallback."

    def test_infer_hard_limit(self, client):
        """Verify strict 413 limit on /infer endpoint."""
        oversized = "x" * 10000
        response = client.post("/infer", json={"prompt": oversized})
        assert response.status_code == 413
        assert "Too long" in response.json()["detail"] or "too long" in response.json()["detail"]

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

    async def test_rag_timeout_handling(self):
        """Verify orchestrator handles RAG timeout gracefully."""
        from app.routing.orchestrator import Orchestrator
        import asyncio
        
        # Patch wait_for to raise TimeoutError
        async def mock_wait_for(coro, timeout):
            raise asyncio.TimeoutError()
            
        with patch("app.routing.orchestrator.asyncio.wait_for", side_effect=mock_wait_for), \
             patch("app.routing.orchestrator.get_semantic_router") as mock_router, \
             patch("app.routing.orchestrator.search_web_context") as mock_search, \
             patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
             
             mock_search.return_value = ("Search completed.", [{"title": "Test", "content": "Context", "url": "url"}])
             
             # Force unknown domain to trigger RAG path
             mock_router_instance = MagicMock()
             mock_router_instance.classify.return_value = ("unknown", 0.2)
             mock_router.return_value = mock_router_instance
             
             orch = Orchestrator()
             result = await orch.route_and_execute("Will this specific RAG query timeout eventually?")
             
             assert result["refused"] is True
             assert "timed out" in result["response"]
