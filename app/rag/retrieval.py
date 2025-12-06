"""
Retrieval Module.
Provides functions to search web context for RAG.
Wraps the Tavily client.
"""

import os
from app.rag.tavily_client import get_tavily_client

def search_web_context(query: str, max_results: int = 3) -> str:
    """
    Search the web for context using Tavily.
    
    Args:
        query: User query
        max_results: Number of results to fetch
        
    Returns:
        Formatted context string
    """
    if not os.getenv("TAVILY_API_KEY"):
        return "Error: TAVILY_API_KEY not configured. Cannot fetch external context."
        
    try:
        tavily = get_tavily_client()
        context = tavily.get_context(query, max_results=max_results)
        return context
    except Exception as e:
        print(f"Retrieval error: {e}")
        return f"Error retrieving context: {str(e)}"
