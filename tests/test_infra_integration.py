import pytest
import os
from unittest.mock import MagicMock, patch, AsyncMock
from app.routing.orchestrator import Orchestrator

# Force deterministic mode
os.environ["USE_DETERMINISTIC_INFERENCE"] = "true"

pytestmark = pytest.mark.anyio

async def test_redis_hit_short_circuits_rag():
    """Verify that a Redis cache hit prevents further processing."""
    
    # Mock DataCollector (Source, since imported locally)
    with patch("app.routing.orchestrator.get_data_collector") as mock_get_collector:
        mock_collector = AsyncMock()
        mock_collector.get_cached_response.return_value = "Cached Response"
        mock_get_collector.return_value = mock_collector
        
        # Patch semantic router (Module level in orchestrator)
        with patch("app.routing.orchestrator.get_semantic_router") as mock_get_router:
                mock_router = MagicMock()
                mock_router.classify.return_value = ("general", 0.5) 
                mock_get_router.return_value = mock_router

                # Execute
                orch = Orchestrator()
                result = await orch.route_and_execute("How to use Redis cache?")
                
                # Verify Redis hit
                assert result["mode"] == "redis_cache"
                assert result["response"] == "Cached Response"

async def test_rag_end_to_end_integration():
    """Verify full RAG flow: Router -> Search -> Reasoner."""
    with patch("app.routing.orchestrator.search_web_context") as mock_search, \
         patch("app.routing.orchestrator.get_semantic_router") as mock_get_router, \
         patch("app.routing.orchestrator.get_reasoner") as mock_get_reasoner, \
         patch("app.routing.orchestrator.get_data_collector") as mock_get_collector, \
         patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
        
        # Router -> unknown
        mock_router = MagicMock()
        mock_router.classify.return_value = ("unknown", 0.2)
        mock_get_router.return_value = mock_router
        
        # Search -> success
        mock_search.return_value = ("Found.", [{"title": "RAG Doc", "content": "RAG Content", "url": "http://rag.com"}])
        
        # Reasoner -> synthesis
        mock_reasoner = AsyncMock()
        mock_reasoner.synthesize_with_context.return_value = "Synthesized Answer."
        mock_get_reasoner.return_value = mock_reasoner

        # Collector -> cache miss
        mock_collector = AsyncMock()
        mock_collector.get_cached_response.return_value = None
        mock_get_collector.return_value = mock_collector
        
        # Execute
        orch = Orchestrator()
        result = await orch.route_and_execute("What is RAG architecture?")
        
        # Verify
        assert result["mode"] == "rag-external"
        assert result["response"] == "Synthesized Answer."
        assert result["citations"][0]["title"] == "RAG Doc"
