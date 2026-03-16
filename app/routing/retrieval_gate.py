"""
Retrieval gate using embedding-based novelty detection.
No heuristics - purely distance from internal knowledge centroid.
"""

import numpy as np
from typing import Optional


class RetrievalGate:
    """
    Determines retrieval eligibility via embedding distance.
    Maintains rolling centroid of confident answers.
    """
    
    def __init__(self, novelty_threshold: float = 0.6):
        import os
        self.use_deterministic = os.getenv("USE_DETERMINISTIC_INFERENCE", "false").lower() == "true"
        
        if not self.use_deterministic:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            self.encoder = None
            
        self.knowledge_centroid: Optional[np.ndarray] = None
        self.confident_embeddings = []
        self.novelty_threshold = novelty_threshold
    
    def update_centroid(self, confident_query: str):
        """
        Update knowledge centroid with new confident query.
        Maintains rolling window of last 100 queries.
        """
        if self.use_deterministic:
            return
            
        emb = self.encoder.encode([confident_query])[0]
        self.confident_embeddings.append(emb)
        
        # Rolling window
        if len(self.confident_embeddings) > 100:
            self.confident_embeddings.pop(0)
        
        # Recompute centroid
        self.knowledge_centroid = np.mean(self.confident_embeddings, axis=0)
        
        print(f"  Centroid updated (n={len(self.confident_embeddings)})")
    
    def should_retrieve(self, query: str) -> bool:
        """
        Single condition: embedding distance from knowledge centroid.
        
        Returns:
            True if query is novel (far from known concepts)
        """
        if self.use_deterministic:
            # Deterministic heuristic: queries > 100 chars are considered 'novel' (complex)
            # This allows testing the 'search' branch without heavy embeddings
            is_novel = len(query) > 100
            print(f"  [Deterministic] Novelty heuristic: len={len(query)} → {'RETRIEVE' if is_novel else 'KNOWN'}")
            return is_novel

        # Cold start: allow retrieval for first queries
        if self.knowledge_centroid is None or len(self.confident_embeddings) < 5:
            print(f"  Cold start: allowing retrieval")
            return True
        
        # Compute distance
        query_emb = self.encoder.encode([query])[0]
        distance = np.linalg.norm(query_emb - self.knowledge_centroid)
        
        is_novel = distance > self.novelty_threshold
        
        print(f"  Novelty distance: {distance:.3f} (threshold: {self.novelty_threshold:.3f}) → {'RETRIEVE' if is_novel else 'KNOWN'}")
        
        return is_novel


# Singleton instance
_gate_instance = None

def get_retrieval_gate() -> RetrievalGate:
    """Get singleton retrieval gate."""
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = RetrievalGate()
    return _gate_instance
