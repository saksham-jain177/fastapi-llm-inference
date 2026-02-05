"""
Retrieval Module.
Provides functions to search web context for RAG.
Wraps the Tavily client.
"""

import os
import re
from app.rag.tavily_client import get_tavily_client

from typing import Tuple, List, Dict

def compute_lexical_relevance(query: str, content: str) -> float:
    """
    Compute a lexical relevance score (0-1) based on keyword overlap.
    No ML, no embeddings. Purely deterministic.
    """
    if not content or not query:
        return 0.0
        
    # Clean and tokenize
    def tokenize(text: str) -> set:
        clean = re.sub(r'[^\w\s]', '', text.lower())
        return set(clean.split())

    query_tokens = tokenize(query)
    content_tokens = tokenize(content)
    
    if not query_tokens:
        return 0.0
        
    # Intersection ratio
    overlap = query_tokens.intersection(content_tokens)
    score = len(overlap) / len(query_tokens)
    
    # Penalize low-diversity "boilerplate" (ratio of unique tokens to total words)
    words = content.lower().split()
    if len(words) > 50:
        diversity = len(set(words)) / len(words)
        if diversity < 0.3:  # high repetition/boilerplate
            score *= 0.5
            
    return min(score, 1.0)

def search_web_context(query: str, max_results: int = 5) -> Tuple[str, List[Dict]]:
    """
    Search the web for context using Tavily.
    Now fetches more results (default 5) for re-ranking.
    """
    if not os.getenv("TAVILY_API_KEY"):
        return "Error: TAVILY_API_KEY not configured. Cannot fetch external context.", []
        
    try:
        tavily = get_tavily_client()
        # Fetch more for re-ranking
        results = tavily.search(query, max_results=max_results, max_retries=3)
        
        if not results:
            return "No relevant information found.", []
            
        return "Search completed.", results
        
    except Exception as e:
        print(f"Retrieval error: {e}")
        return f"Error retrieving context: {str(e)}", []
