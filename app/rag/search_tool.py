"""
SearchTool: Encapsulates web search functionality using Tavily.
Provides simple interfaces for the Orchestrator to solve 'dumb' hallucinations.
"""
from typing import Dict, Any, Optional
from app.rag.tavily_client import get_tavily_client

class SearchTool:
    def __init__(self):
        self.client = get_tavily_client()
        
    def strict_search(self, query: str) -> Dict[str, Any]:
        """
        Execute a strict search. Returns structured data.
        Used when precise factual info is required.
        """
        # Increase results for better coverage
        results = self.client.search(query, max_results=5)
        
        if not results:
             return {
                 "found": False,
                 "context": "No relevant information found via external search.",
                 "results": []
             }
             
        # Format context for RAG
        # We reuse the client's formatting logic but ensure we got it
        context_str = self.client.get_context(query, max_results=5)
        
        return {
            "found": True,
            "context": context_str,
            "results": results,
            "top_url": results[0]["url"] if results else None
        }

# Global instance
_search_tool_instance = None

def get_search_tool():
    global _search_tool_instance
    if _search_tool_instance is None:
        _search_tool_instance = SearchTool()
    return _search_tool_instance
