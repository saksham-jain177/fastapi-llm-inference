"""
Retrieval Module.
Provides functions to search web context for RAG.
Wraps the Tavily client.
"""

import os
from app.rag.tavily_client import get_tavily_client

from typing import Tuple, List, Dict

def search_web_context(query: str, max_results: int = 3) -> Tuple[str, List[Dict]]:
    """
    Search the web for context using Tavily.
    
    Args:
        query: User query
        max_results: Number of results to fetch
        
    Returns:
        Tuple[context_str, results_list]
    """
    if not os.getenv("TAVILY_API_KEY"):
        return "Error: TAVILY_API_KEY not configured. Cannot fetch external context.", []
        
    try:
        tavily = get_tavily_client()
        # Get raw results to preserve metadata
        results = tavily.search(query, max_results=max_results, max_retries=3)
        
        if not results:
            return "No relevant information found.", []
            
        # Format context string (replicating get_context logic)
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"Source {i}: {result['title']}\n{result['content']}\nURL: {result['url']}"
            )
        
        return "\n\n".join(context_parts), results
        
    except Exception as e:
        print(f"Retrieval error: {e}")
        return f"Error retrieving context: {str(e)}", []
