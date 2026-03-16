"""
Semantic query classifier using sentence transformers.
Routes queries to appropriate domain-specific adapters.
"""

import numpy as np
from typing import Tuple, List
from pathlib import Path


class SemanticRouter:
    """
    Classifies queries into domains using semantic similarity.
    Uses all-MiniLM-L6-v2 for fast, efficient embeddings.
    """
    
    # Domain exemplars (representative queries for each domain)
    DOMAIN_EXEMPLARS = {
        "code": [
            "Write a Python function to sort a list",
            "How do I reverse a string in JavaScript",
            "Create a REST API endpoint",
            "Debug this SQL query",
            "Implement binary search algorithm"
        ],
        "medical": [
            "What are the symptoms of diabetes",
            "Explain the cardiovascular system",
            "Treatment for hypertension",
            "Side effects of antibiotics",
            "Diagnosis of common cold"
        ],
        "legal": [
            "What is contract law",
            "Explain intellectual property rights",
            "Terms of service requirements",
            "Privacy policy compliance",
            "Employment law regulations"
        ],
        "general": [
            "What is the weather like",
            "Tell me about history",
            "Explain quantum physics",
            "How does photosynthesis work",
            "What is artificial intelligence"
        ]
    }
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize semantic router with sentence transformer model."""
        import os
        self.use_deterministic = os.getenv("USE_DETERMINISTIC_INFERENCE", "false").lower() == "true"
        
        if self.use_deterministic:
            print("[Deterministic Mode] Semantic router using keyword-based routing (Offline)")
            self.model = None
            self.domain_exemplar_embeddings = {}
            return

        from sentence_transformers import SentenceTransformer
        print(f"Loading semantic router: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Store individual exemplar embeddings (not averaged)
        self.domain_exemplar_embeddings = {}
        for domain, exemplars in self.DOMAIN_EXEMPLARS.items():
            # Encode all exemplars for this domain
            embeddings = self.model.encode(exemplars)
            self.domain_exemplar_embeddings[domain] = embeddings
        
        print(f"Semantic router loaded with {len(self.domain_exemplar_embeddings)} domains")
    
    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify query into a domain using MAX similarity to any exemplar.
        
        Args:
            query: User query to classify
            
        Returns:
            Tuple of (domain, confidence_score)
        """
        if self.use_deterministic:
            # Keyword-based routing for CI
            query_lower = query.lower()
            if "python" in query_lower or "def " in query_lower or "code" in query_lower:
                return "code", 1.0
            if "patient" in query_lower or "symptoms" in query_lower or "medical" in query_lower:
                return "medical", 1.0
            if "law" in query_lower or "legal" in query_lower or "contract" in query_lower:
                return "legal", 1.0
            # Strict Truth: If we don't know the domain via strict keyword, we don't guess.
            return "unknown", 0.0

        # Encode query
        query_embedding = self.model.encode([query])[0]
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Compute MAX similarity to each domain
        similarities = {}
        for domain, exemplar_embeddings in self.domain_exemplar_embeddings.items():
            # Get similarity to each exemplar, take the MAX
            domain_similarities = []
            for exemplar_emb in exemplar_embeddings:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    exemplar_emb.reshape(1, -1)
                )[0][0]
                domain_similarities.append(float(similarity))
            
            # Use MAX similarity (query matches at least one exemplar well)
            similarities[domain] = max(domain_similarities)
        
        # Get best match
        best_domain = max(similarities, key=similarities.get)
        confidence = similarities[best_domain]

        # ENFORCE CONFIDENCE GATE
        # If signal is too low, treat as 'unknown' to force RAG/Refusal
        ROUTER_CONFIDENCE_THRESHOLD = 0.35
        if confidence < ROUTER_CONFIDENCE_THRESHOLD:
            print(f"[Semantic Router] Confidence {confidence:.3f} below threshold {ROUTER_CONFIDENCE_THRESHOLD}")
            best_domain = "unknown"
        
        # Debug logging
        print(f"[Semantic Router] Query: '{query[:50]}...'")
        print(f"[Semantic Router] Similarities: {similarities}")
        print(f"[Semantic Router] Best: {best_domain} ({confidence:.3f})")
        
        return best_domain, confidence
    
    def get_top_domains(self, query: str, top_k: int = 2) -> List[Tuple[str, float]]:
        """
        Get top-k most likely domains for a query.
        """
        if self.use_deterministic:
            domain, conf = self.classify(query)
            return [(domain, conf)]

        query_embedding = self.model.encode([query])[0]
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = {}
        # Fixed: Use domain_exemplar_embeddings and take MAX for consistency with classify()
        for domain, exemplar_embeddings in self.domain_exemplar_embeddings.items():
            domain_similarities = []
            for exemplar_emb in exemplar_embeddings:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    exemplar_emb.reshape(1, -1)
                )[0][0]
                domain_similarities.append(float(similarity))
            similarities[domain] = max(domain_similarities)
        
        # Sort by similarity
        sorted_domains = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_domains[:top_k]


# Global instance and lock
import threading
_semantic_router_lock = threading.Lock()
_semantic_router = None


def get_semantic_router() -> SemanticRouter:
    """Get or create semantic router instance."""
    global _semantic_router
    if _semantic_router is None:
        with _semantic_router_lock:
            if _semantic_router is None:
                _semantic_router = SemanticRouter()
    return _semantic_router

def reset_semantic_router():
    """Reset the singleton instance."""
    global _semantic_router
    with _semantic_router_lock:
        _semantic_router = None
