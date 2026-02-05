import pytest
import os
from unittest.mock import MagicMock, patch

# Ensure we use deterministic mode for tests
os.environ["USE_DETERMINISTIC_INFERENCE"] = "true"

pytestmark = pytest.mark.anyio

class TestRoutingLogic:
    """
    Tests for routing invariants (Orchestrator).
    Uses hardware-independent mode to avoid needing real GPUs or LLMs.
    """

    async def test_orchestrator_routing_rag(self):
        """Test that Orchestrator correctly calls RAG path for unknown domain."""
        from app.routing.orchestrator import Orchestrator
        from unittest.mock import AsyncMock
        
        # We patch the specific dependencies of the orchestrator
        with patch("app.routing.orchestrator.search_web_context") as mock_search, \
             patch("app.routing.orchestrator.get_semantic_router") as mock_get_router, \
             patch("app.routing.orchestrator.get_reasoner") as mock_get_reasoner, \
             patch("app.routing.orchestrator.get_data_collector") as mock_get_collector, \
             patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            
            # Setup router mock to return unknown domain
            mock_router = MagicMock()
            mock_router.classify.return_value = ("unknown", 0.2)
            mock_get_router.return_value = mock_router
            
            # Setup search mock
            mock_search.return_value = ("Search completed.", [{"title": "Test Source", "content": "Mock Context", "url": "http://test.com"}])
            
            # Setup reasoner mock
            mock_reasoner_instance = AsyncMock()
            mock_reasoner_instance.synthesize_with_context.return_value = "RAG Response."
            mock_get_reasoner.return_value = mock_reasoner_instance

            # Setup collector mock (ensure cache miss)
            mock_collector = AsyncMock()
            mock_collector.get_cached_response.return_value = None
            mock_get_collector.return_value = mock_collector
            
            # Execute
            orch = Orchestrator()
            result = await orch.route_and_execute("What is the weather?")
            
            # Verify
            assert result["mode"] == "rag-external"
            assert result["response"] == "RAG Response."
            # Citations include content now as they are passed through from packer
            assert result["citations"][0]["title"] == "Test Source"
            assert result["citations"][0]["url"] == "http://test.com"
            assert result["citations"][0]["content"] == "Mock Context"
            mock_search.assert_called_once_with("What is the weather?")
            mock_reasoner_instance.synthesize_with_context.assert_called_once()
            mock_collector.log_interaction.assert_called_once()

    async def test_orchestrator_detects_truncated_response(self):
        """Test that truncated answers get a warning appended."""
        from app.routing.orchestrator import Orchestrator
        from unittest.mock import AsyncMock
        with patch("app.routing.orchestrator.search_web_context") as mock_search, \
             patch("app.routing.orchestrator.get_semantic_router") as mock_get_router, \
             patch("app.routing.orchestrator.get_reasoner") as mock_get_reasoner, \
             patch("app.routing.orchestrator.get_data_collector") as mock_get_collector, \
             patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            
            # Setup router to return unknown
            mock_router = MagicMock()
            mock_router.classify.return_value = ("unknown", 0.2)
            mock_get_router.return_value = mock_router
            
            # Setup search (tuples)
            mock_search.return_value = ("Context", [])
            
            # Setup reasoner to return incomplete text
            mock_reasoner = AsyncMock()
            mock_reasoner.synthesize_with_context.return_value = "The capital of France is Paris and"  # No punctuation, ends with stop word
            mock_get_reasoner.return_value = mock_reasoner
            
            # Setup collector mock (ensure cache miss)
            mock_collector = AsyncMock()
            mock_collector.get_cached_response.return_value = None
            mock_get_collector.return_value = mock_collector

            orch = Orchestrator()
            result = await orch.route_and_execute("Query")
            
            # Verify warning appended
            assert "(This answer may be incomplete" in result["response"]
            assert result["response"].startswith("The capital of France is Paris and")

    async def test_orchestrator_strips_hallucinated_citations(self):
        """Test that [Source X] markers are stripped if no citations exist."""
        from app.routing.orchestrator import Orchestrator
        from unittest.mock import AsyncMock
        with patch("app.routing.orchestrator.search_web_context") as mock_search, \
             patch("app.routing.orchestrator.get_semantic_router") as mock_get_router, \
             patch("app.routing.orchestrator.get_reasoner") as mock_get_reasoner, \
             patch("app.routing.orchestrator.get_data_collector") as mock_get_collector, \
             patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            
            # Setup router to return unknown
            mock_router = MagicMock()
            mock_router.classify.return_value = ("unknown", 0.2)
            mock_get_router.return_value = mock_router
            
            # Setup search (Returns NO citations)
            mock_search.return_value = ("Search completed.", [])
            
            # Setup reasoner to hallucinates sources
            mock_reasoner = AsyncMock()
            mock_reasoner.synthesize_with_context.return_value = "This is a fact [Source 1] and another [Source 2]."
            mock_get_reasoner.return_value = mock_reasoner
            
            # Setup collector mock (ensure cache miss)
            mock_collector = AsyncMock()
            mock_collector.get_cached_response.return_value = None
            mock_get_collector.return_value = mock_collector

            orch = Orchestrator()
            result = await orch.route_and_execute("Query")
            
            # Verify strip
            assert "[Source 1]" not in result["response"]
            assert "[Source 2]" not in result["response"]
            # Extra space preserved from re.sub
            assert result["response"] == "This is a fact  and another ."

    async def test_orchestrator_routes_ambiguous_query_to_rag(self):
        """Verify that a query below the semantic threshold routes to RAG."""
        from app.routing.orchestrator import Orchestrator
        from unittest.mock import AsyncMock, patch, MagicMock

        query = "Some completely ambiguous query about PageIndex"
        
        with patch("app.routing.orchestrator.get_semantic_router") as mock_get_router, \
             patch("app.routing.orchestrator.get_data_collector") as mock_get_collector, \
             patch("app.routing.orchestrator.search_web_context") as mock_search, \
             patch("app.routing.orchestrator.get_reasoner") as mock_get_reasoner, \
             patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            
            # 1. Setup Router to return 'unknown'
            mock_router = MagicMock()
            mock_router.classify.return_value = ("unknown", 0.15)
            mock_get_router.return_value = mock_router
            
            # 2. Setup Collector
            mock_collector = AsyncMock()
            mock_collector.get_cached_response.return_value = None
            mock_get_collector.return_value = mock_collector
            
            # 3. Setup RAG Search & Reasoner
            mock_search.return_value = ("Search completed.", [{"title": "Source 1", "content": "PageIndex is a C# concept", "url": "http://test.com"}])
            mock_reasoner = AsyncMock()
            mock_reasoner.synthesize_with_context.return_value = "PageIndex is found in pagination."
            mock_get_reasoner.return_value = mock_reasoner
            
            orch = Orchestrator()
            result = await orch.route_and_execute(query)
            
            # 4. Verify RAG was triggered
            assert result["mode"] == "rag-external"
            assert "pagination" in result["response"]
            mock_search.assert_called_once_with(query)
            mock_reasoner.synthesize_with_context.assert_called_once()
