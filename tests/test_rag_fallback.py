import pytest
import os
from unittest.mock import MagicMock, patch
from app.rag.retrieval import search_web_context

def test_tavily_success_no_fallback():
    """Verify standard path uses Tavily and does not fallback."""
    with patch("app.rag.retrieval.get_tavily_client") as mock_get_tavily, \
         patch("app.rag.duckduckgo_client.get_ddg_client") as mock_get_ddg, \
         patch.dict(os.environ, {"TAVILY_API_KEY": "test"}):
        
        # Setup Tavily success
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = [{"title": "Tavily Result", "url": "http://tavily.com", "content": "Success"}]
        mock_get_tavily.return_value = mock_tavily
        
        status, results = search_web_context("query")
        
        assert "Search completed" in status
        assert len(results) == 1
        assert results[0]["title"] == "Tavily Result"
        
        # Verify fallback NOT called
        mock_get_ddg.assert_not_called()

def test_tavily_failure_triggers_fallback():
    """Verify fallback when Tavily raises exception."""
    with patch("app.rag.retrieval.get_tavily_client") as mock_get_tavily, \
         patch("app.rag.duckduckgo_client.get_ddg_client") as mock_get_ddg, \
         patch.dict(os.environ, {"TAVILY_API_KEY": "test"}):
        
        # Setup Tavily failure
        mock_tavily = MagicMock()
        mock_tavily.search.side_effect = Exception("API Down")
        mock_get_tavily.return_value = mock_tavily
        
        # Setup DDG success
        mock_ddg = MagicMock()
        mock_ddg.search.return_value = [{"title": "DDG Result", "url": "http://ddg.com", "content": "Fallback", "source": "duckduckgo"}]
        mock_get_ddg.return_value = mock_ddg
        
        status, results = search_web_context("query")
        
        assert "Fallback" in status
        assert len(results) == 1
        assert results[0]["source"] == "duckduckgo"
        
        # Verify both called
        mock_tavily.search.assert_called_once()
        mock_ddg.search.assert_called_once()

def test_tavily_empty_triggers_fallback():
    """Verify fallback when Tavily returns empty list."""
    with patch("app.rag.retrieval.get_tavily_client") as mock_get_tavily, \
         patch("app.rag.duckduckgo_client.get_ddg_client") as mock_get_ddg, \
         patch.dict(os.environ, {"TAVILY_API_KEY": "test"}):
        
        # Setup Tavily empty
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = []
        mock_get_tavily.return_value = mock_tavily
        
        # Setup DDG success
        mock_ddg = MagicMock()
        mock_ddg.search.return_value = [{"title": "DDG Result", "url": "http://ddg.com", "content": "Fallback"}]
        mock_get_ddg.return_value = mock_ddg
        
        status, results = search_web_context("query")
        
        assert "Fallback" in status
        assert len(results) == 1
        mock_ddg.search.assert_called_once()

def test_both_fail_gracefully():
    """Verify empty return when both fail."""
    with patch("app.rag.retrieval.get_tavily_client") as mock_get_tavily, \
         patch("app.rag.duckduckgo_client.get_ddg_client") as mock_get_ddg, \
         patch.dict(os.environ, {"TAVILY_API_KEY": "test"}):
        
        mock_tavily = MagicMock()
        mock_tavily.search.side_effect = Exception("Tavily Down")
        mock_get_tavily.return_value = mock_tavily
        
        mock_ddg = MagicMock()
        mock_ddg.search.side_effect = Exception("DDG Down")
        mock_get_ddg.return_value = mock_ddg
        
        status, results = search_web_context("query")
        
        assert "No relevant information" in status
        assert results == []
