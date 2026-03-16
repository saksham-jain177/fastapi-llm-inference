"""
Context Manager for RAG.
Handles ranking, token budgeting, and citation-aware packing.
"""

import os
import threading
from typing import List, Dict, Tuple
from app.rag.retrieval import compute_lexical_relevance

class ContextPacker:
    """
    Ranks search results and packs them into a deterministic context string
    within a specified token budget. Supports Cross-Encoder reranking.
    """
    
    def __init__(self, token_budget: int = 1500):
        self.token_budget = token_budget
        self.use_deterministic = os.getenv("USE_DETERMINISTIC_INFERENCE", "false").lower() == "true"
        self.encoder = None
        self.tokenizer = None
        
        if not self.use_deterministic:
            try:
                from sentence_transformers import CrossEncoder
                from transformers import AutoTokenizer
                print("Loading Cross-Encoder and Tokenizer for RAG...")
                self.encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            except ImportError:
                print("Warning: sentence_transformers/transformers not installed. Falling back to deterministic.")
                self.use_deterministic = True

    def _get_token_count(self, text: str) -> int:
        if self.use_deterministic or self.tokenizer is None:
            return max(1, len(text) // 4)
        return len(self.tokenizer.encode(text))

    def pack(self, query: str, results: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Rank results using Cross-Encoder (or lexical fallback) and pack into token budget.
        """
        if not results:
            return "No relevant context available.", []

        # 1. Rank results
        scored_results = []
        if self.use_deterministic or self.encoder is None:
            for res in results:
                score = compute_lexical_relevance(query, res.get('content', ''))
                scored_results.append((score, res))
        else:
            pairs = [[query, res.get('content', '')] for res in results]
            scores = self.encoder.predict(pairs)
            for score, res in zip(scores, results):
                scored_results.append((float(score), res))
            
        # Sort by score descending
        ranked = sorted(scored_results, key=lambda x: x[0], reverse=True)
        
        # 2. Pack within budget
        packed_parts = []
        used_citations = []
        current_tokens = 0
        
        for i, (score, result) in enumerate(ranked):
            source_id = i + 1
            snippet = f"Source {source_id}: {result.get('title', 'Unknown')}\n{result.get('content', '')}\nURL: {result.get('url', '')}"
            snippet_tokens = self._get_token_count(snippet)
            
            if current_tokens + snippet_tokens > self.token_budget:
                if not packed_parts:
                    # Truncate first result if it alone exceeds budget
                    allowed_tokens = max(10, self.token_budget - current_tokens - 10) # 10 tokens margin for URL etc
                    content = result.get('content', '')
                    allowed_chars = allowed_tokens * 4
                    truncated = content[:allowed_chars] + "..."
                    snippet = f"Source {source_id}: {result.get('title', 'Unknown')}\n{truncated}\nURL: {result.get('url', '')}"
                    packed_parts.append(snippet)
                    used_citations.append(result)
                break
            
            packed_parts.append(snippet)
            used_citations.append(result)
            current_tokens += snippet_tokens + 2 # extra for newlines
            
        final_context = "\n\n".join(packed_parts)
        return final_context, used_citations


_packer_lock = threading.Lock()
_packer = None

def get_context_packer() -> ContextPacker:
    global _packer
    if _packer is None:
        with _packer_lock:
            if _packer is None:
                _packer = ContextPacker()
    return _packer
