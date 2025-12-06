"""
Semantic query classifier using sentence transformers.
Routes queries to appropriate domain-specific adapters.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
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
        
        # Debug logging
        print(f"[Semantic Router] Query: '{query[:50]}...'")
        print(f"[Semantic Router] Similarities: {similarities}")
        print(f"[Semantic Router] Best: {best_domain} ({confidence:.3f})")
        
        return best_domain, confidence
    
    def get_top_domains(self, query: str, top_k: int = 2) -> List[Tuple[str, float]]:
        """
        Get top-k most likely domains for a query.
        
        Args:
            query: User query
            top_k: Number of top domains to return
            
        Returns:
            List of (domain, confidence) tuples, sorted by confidence
        """
        query_embedding = self.model.encode([query])[0]
        
        similarities = {}
        for domain, domain_embedding in self.domain_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                domain_embedding.reshape(1, -1)
            )[0][0]
            similarities[domain] = float(similarity)
        
        # Sort by similarity
        sorted_domains = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_domains[:top_k]


# Global instance
_semantic_router = None


def get_semantic_router() -> SemanticRouter:
    """Get or create semantic router instance."""
    global _semantic_router
    if _semantic_router is None:
        _semantic_router = SemanticRouter()
    return _semantic_router
