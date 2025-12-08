
import pytest
from unittest.mock import MagicMock, patch
from app.routing.query_analyzer import QueryAnalyzer, QueryFeatures

class TestRoutingLogic:
    """
    Tests for the Agentic RAG logic (QueryAnalyzer -> Orchestrator).
    Uses mocks to avoid needing real GPUs or LLMs.
    """

    @pytest.fixture
    def mock_analyzer(self):
        analyzer = QueryAnalyzer()
        # Mock the expensive components
        analyzer.llm_judge = MagicMock()
        return analyzer

    def test_analyzer_detects_code(self, mock_analyzer):
        """Test that code keywords trigger code domain."""
        query = "Write a python function to sort a list by name"
        features = mock_analyzer.extract_features(query)
        
        # extract_features returns the DOMAIN keys (e.g., "code"), not the markers
        assert "code" in features.domain_markers

    def test_analyzer_detects_realtime(self, mock_analyzer):
        """Test that 'news' triggers realtime/external strategy."""
        query = "What is the latest news about AI?"
        
        # We must mock the internal logic because extract_features is deterministic
        # but analyze() uses rule engine. 
        # Actually, QueryAnalyzer._rule_engine_classify uses features directly.
        
        analysis = mock_analyzer.analyze(query)
        # Should be classified as external_search due to "latest" and "news"
        assert analysis["intent"] == "external_search"

    def test_analyzer_detects_complex_reasoning(self, mock_analyzer):
        """Test that complex questions trigger reasoning."""
        query = "Analyze the impact of interest rates and explain step-by-step."
        
        analysis = mock_analyzer.analyze(query)
        assert analysis["intent"] == "complex_reasoning"

    def test_orchestrator_routing_rag(self):
        """Test that Orchestrator correctly calls RAG path."""
        import sys
        
        # Mock modules that might trigger heavy imports (torch, etc.)
        sys.modules["app.models.quantized"] = MagicMock()
        sys.modules["app.models.adapter_manager"] = MagicMock()
        sys.modules["app.rag.retrieval"] = MagicMock()
        
        # Now import orchestrator
        import app.routing.orchestrator
        from app.routing.orchestrator import Orchestrator
        
        # We still need to patch the specific functions used inside orchestrator
        # Because we mocked the whole module, we can just configure the mock in sys.modules
        # OR continued to use patch. Using patch is safer for specific targets.
        
        with patch("app.routing.orchestrator.search_web_context") as mock_search, \
             patch("app.routing.orchestrator.get_query_analyzer") as mock_get_analyzer, \
             patch("app.routing.orchestrator.get_ollama_reasoner") as mock_get_reasoner, \
             patch("app.routing.orchestrator.get_adapter_manager"):
            
            # Setup mocks
            mock_analyzer_instance = MagicMock()
            mock_analyzer_instance.analyze.return_value = {"intent": "external_search"}
            mock_get_analyzer.return_value = mock_analyzer_instance
            
            mock_search.return_value = "Mock Context"
            
            mock_reasoner_instance = MagicMock()
            mock_reasoner_instance.synthesize_with_context.return_value = "RAG Response"
            mock_get_reasoner.return_value = mock_reasoner_instance
            
            # Execute
            orch = Orchestrator()
            result = orch.route_and_execute("What is the weather?")
            
            # Verify
            assert result["mode"] == "rag-external"
            mock_search.assert_called_once()
            mock_reasoner_instance.synthesize_with_context.assert_called_once()
