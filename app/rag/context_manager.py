"""
Context Manager for RAG.
Handles ranking, token budgeting, and citation-aware packing.
"""

import re
from typing import List, Dict, Tuple
from app.rag.retrieval import compute_lexical_relevance

class ContextPacker:
    """
    Ranks search results and packs them into a deterministic context string
    within a specified token (char-based) budget.
    """
    
    def __init__(self, char_budget: int = 6000):
        self.char_budget = char_budget

    def pack(self, query: str, results: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Rank results by lexical relevance and pack into budget.
        
        Returns:
            Tuple[packed_context_string, subset_of_results_used]
        """
        if not results:
            return "No relevant context available.", []

        # 1. Rank by lexical relevance
        scored_results = []
        for res in results:
            score = compute_lexical_relevance(query, res.get('content', ''))
            scored_results.append((score, res))
            
        # Sort by score descending
        ranked = sorted(scored_results, key=lambda x: x[0], reverse=True)
        
        # 2. Pack within budget
        packed_parts = []
        used_citations = []
        current_chars = 0
        
        # We always try to include at least the top relevance result
        for i, (score, result) in enumerate(ranked):
            # Format snippet
            source_id = i + 1
            snippet = f"Source {source_id}: {result.get('title', 'Unknown')}\n{result.get('content', '')}\nURL: {result.get('url', '')}"
            
            snippet_len = len(snippet)
            
            # Check budget
            if current_chars + snippet_len > self.char_budget:
                # If this is the FIRST result and it's too big, we MUST truncate it
                if not packed_parts:
                    allowed = self.char_budget - current_chars - 50 # margin
                    truncated_content = result.get('content', '')[:allowed] + "..."
                    snippet = f"Source {source_id}: {result.get('title', 'Unknown')}\n{truncated_content}\nURL: {result.get('url', '')}"
                    packed_parts.append(snippet)
                    used_citations.append(result)
                break # stop adding more sources
            
            packed_parts.append(snippet)
            used_citations.append(result)
            current_chars += snippet_len + 2 # extra for newlines
            
        final_context = "\n\n".join(packed_parts)
        return final_context, used_citations

def get_context_packer() -> ContextPacker:
    return ContextPacker()
