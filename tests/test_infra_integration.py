import pytest
import os
from unittest.mock import MagicMock, patch, AsyncMock
from app.routing.orchestrator import Orchestrator

# Force deterministic mode
os.environ["USE_DETERMINISTIC_INFERENCE"] = "true"

pytestmark = pytest.mark.anyio

async def test_redis_hit_short_circuits_mongodb():
    """Verify that a Redis cache hit prevents checking MongoDB/Chroma."""
    
    # Mock DataCollector (Source, since imported locally)
    with patch("app.routing.orchestrator.get_data_collector") as mock_get_collector:
        mock_collector = AsyncMock()
        mock_collector.get_cached_response.return_value = "Cached Response"
        mock_get_collector.return_value = mock_collector
        
        # Mock FeedbackRetriever (Source, since imported locally)
        with patch("app.routing.orchestrator.get_feedback_retriever") as mock_get_retriever:
            mock_retriever = MagicMock()
            mock_get_retriever.return_value = mock_retriever
            
            # Patch analyzer (Module level in orchestrator)
            with patch("app.routing.orchestrator.get_query_analyzer") as mock_get_analyzer:
                mock_analyzer = MagicMock()
                mock_analyzer.analyze.return_value = {"intent": "simple_internal"} 
                mock_get_analyzer.return_value = mock_analyzer

                # Execute
                orch = Orchestrator()
                result = await orch.route_and_execute("Cached query")
                
                # Verify Redis hit
                assert result["mode"] == "redis_cache"
                assert result["response"] == "Cached Response"
                
                # Verify Mongo/Chroma NOT accessed
                mock_retriever.search_similar.assert_not_called()

async def test_mongodb_hit_short_circuits_rag():
    """Verify that a MongoDB/Chroma hit (when Redis misses) prevents external RAG."""
    
    # Mock DataCollector (Source)
    with patch("app.routing.orchestrator.get_data_collector") as mock_get_collector:
        mock_collector = AsyncMock()
        mock_collector.get_cached_response.return_value = None # Redis Miss
        mock_collector.cache_response = AsyncMock()
        mock_get_collector.return_value = mock_collector
        
        # Mock FeedbackRetriever for a Hit (Source)
        with patch("app.routing.orchestrator.get_feedback_retriever") as mock_get_retriever:
            mock_retriever = MagicMock()
            # Return high confidence match
            mock_retriever.search_similar.return_value = [{
                "response": "Memory Response",
                "similarity": 0.95,
                "confidence": 0.95,
                "query": "Previous query"
            }]
            mock_get_retriever.return_value = mock_retriever
            
            # Mock Analyzer (Module level)
            with patch("app.routing.orchestrator.get_query_analyzer") as mock_get_analyzer:
                mock_analyzer = MagicMock()
                mock_analyzer.analyze.return_value = {"intent": "external_search"}
                mock_get_analyzer.return_value = mock_analyzer
                
                # Mock RAG search (Module level)
                with patch("app.routing.orchestrator.search_web_context") as mock_search:
                    mock_search.return_value = ("Context", [])
                    
                    # Mock reasoner (now async)
                    with patch("app.routing.orchestrator.get_reasoner") as mock_get_reasoner:
                        mock_reasoner = AsyncMock()
                        mock_reasoner.synthesize_with_context.return_value = "RAG Response"
                        mock_get_reasoner.return_value = mock_reasoner

                        # Execute
                        orch = Orchestrator()
                        result = await orch.route_and_execute("Memory query")
                        
                        # Verify Memory hit
                        assert result["mode"] == "memory"
                        assert result["response"] == "Memory Response"
                        
                        # Verify RAG NOT triggered
                        mock_search.assert_not_called()
                        
                        # Verify we tried to cache the result
                        mock_collector.cache_response.assert_called()
