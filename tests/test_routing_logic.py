import pytest
import os
from unittest.mock import MagicMock, patch
from app.routing.query_analyzer import QueryAnalyzer, QueryFeatures
# Orchestrator imported inside tests to fix patching issues

# Ensure we use deterministic mode for tests
os.environ["USE_DETERMINISTIC_INFERENCE"] = "true"

class TestRoutingLogic:
    """
    Tests for the Agentic RAG logic (QueryAnalyzer -> Orchestrator).
    Uses hardware-independent mode to avoid needing real GPUs or LLMs.
    """

    @pytest.fixture
    def mock_analyzer(self):
        analyzer = QueryAnalyzer()
        # Mock the judge to avoid hitting the actual backend (even deterministic) if needed,
        # but here we can just use the real analyzer if we want.
        # However, for control, we'll mock the judge.
        analyzer.judge = MagicMock()
        return analyzer

    def test_analyzer_detects_code(self, mock_analyzer):
        """Test that code keywords trigger code domain."""
        query = "Write a python function to sort a list by name"
        features = mock_analyzer.extract_features(query)
        assert "code" in features.domain_markers

    def test_analyzer_detects_realtime(self, mock_analyzer):
        """Test that 'news' triggers realtime/external strategy."""
        query = "What is the latest news about AI?"
        analysis = mock_analyzer.analyze(query)
        assert analysis["intent"] == "external_search"

    def test_analyzer_detects_complex_reasoning(self, mock_analyzer):
        """Test that complex questions trigger reasoning."""
        query = "Analyze the impact of interest rates and explain step-by-step."
        analysis = mock_analyzer.analyze(query)
        assert analysis["intent"] == "complex_reasoning"

    @pytest.mark.asyncio
    async def test_orchestrator_routing_rag(self):
        """Test that Orchestrator correctly calls RAG path."""
        from app.routing.orchestrator import Orchestrator
        # We patch the specific dependencies of the orchestrator
        with patch("app.routing.orchestrator.search_web_context") as mock_search, \
             patch("app.routing.orchestrator.get_query_analyzer") as mock_get_analyzer, \
             patch("app.routing.orchestrator.get_reasoner") as mock_get_reasoner, \
             patch("app.routing.orchestrator.asyncio.create_task") as mock_create_task, \
             patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            
            # Setup analyzer mock
            mock_analyzer_instance = MagicMock()
            mock_analyzer_instance.analyze.return_value = {"intent": "external_search"}
            mock_get_analyzer.return_value = mock_analyzer_instance
            
            # Setup search mock
            mock_search.return_value = ("Mock Context", [{"title": "Test Source", "url": "http://test.com"}])
            
            # Setup reasoner mock
            mock_reasoner_instance = MagicMock()
            mock_reasoner_instance.synthesize_with_context.return_value = "RAG Response."
            mock_get_reasoner.return_value = mock_reasoner_instance
            
            # Execute
            orch = Orchestrator()
            result = await orch.route_and_execute("What is the weather?")
            
            # Verify
            assert result["mode"] == "rag-external"
            assert result["response"] == "RAG Response."
            assert result["citations"] == [{"title": "Test Source", "url": "http://test.com"}]
            mock_search.assert_called_once_with("What is the weather?")
            mock_reasoner_instance.synthesize_with_context.assert_called_once()
            
            # Verify and close the coroutine to avoid RuntimeWarning
            mock_create_task.assert_called_once()
            coro = mock_create_task.call_args[0][0]
            coro.close()

    @pytest.mark.asyncio
    async def test_orchestrator_detects_truncated_response(self):
        """Test that truncated answers get a warning appended."""
        from app.routing.orchestrator import Orchestrator
        with patch("app.routing.orchestrator.search_web_context") as mock_search, \
             patch("app.routing.orchestrator.get_query_analyzer") as mock_get_analyzer, \
             patch("app.routing.orchestrator.get_reasoner") as mock_get_reasoner, \
             patch("app.routing.orchestrator.asyncio.create_task") as mock_create_task, \
             patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            
            # Setup intent
            mock_analyzer = MagicMock()
            mock_analyzer.analyze.return_value = {"intent": "external_search"}
            mock_get_analyzer.return_value = mock_analyzer
            
            # Setup search (tuples)
            mock_search.return_value = ("Context", [])
            
            # Setup reasoner to return incomplete text
            mock_reasoner = MagicMock()
            mock_reasoner.synthesize_with_context.return_value = "The capital of France is Paris and"  # No punctuation, ends with stop word
            mock_get_reasoner.return_value = mock_reasoner
            
            orch = Orchestrator()
            result = await orch.route_and_execute("Query")
            
            # Verify warning appended
            assert "(This answer may be incomplete" in result["response"]
            assert result["response"].startswith("The capital of France is Paris and")

    @pytest.mark.asyncio
    async def test_orchestrator_strips_hallucinated_citations(self):
        """Test that [Source X] markers are stripped if no citations exist."""
        from app.routing.orchestrator import Orchestrator
        with patch("app.routing.orchestrator.search_web_context") as mock_search, \
             patch("app.routing.orchestrator.get_query_analyzer") as mock_get_analyzer, \
             patch("app.routing.orchestrator.get_reasoner") as mock_get_reasoner, \
             patch("app.routing.orchestrator.asyncio.create_task") as mock_create_task, \
             patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            
            # Setup intent
            mock_analyzer = MagicMock()
            mock_analyzer.analyze.return_value = {"intent": "external_search"}
            mock_get_analyzer.return_value = mock_analyzer
            
            # Setup search (Returns NO citations)
            mock_search.return_value = ("Context", [])
            
            # Setup reasoner to hallucinates sources
            mock_reasoner = MagicMock()
            mock_reasoner.synthesize_with_context.return_value = "This is a fact [Source 1] and another [Source 2]."
            mock_get_reasoner.return_value = mock_reasoner
            
            orch = Orchestrator()
            result = await orch.route_and_execute("Query")
            
            # Verify strip
            assert "[Source 1]" not in result["response"]
            assert "[Source 2]" not in result["response"]
            assert result["response"] == "This is a fact  and another ."
