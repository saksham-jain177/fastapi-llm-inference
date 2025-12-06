"""
Ollama-based LLM judge for domain classification.
Uses local or cloud Ollama for more accurate classification than semantic router.
"""

import ollama
import os
from typing import Tuple


class OllamaJudge:
    """
    Uses Ollama LLM to classify queries when semantic router confidence is low.
    More accurate but slower than semantic routing.
    """
    
    CLASSIFICATION_PROMPT = """You are a domain classifier. Classify the following query into EXACTLY ONE category.

Categories:
- code: Programming, software, algorithms, debugging, API development
- medical: Health, medicine, biology, symptoms, treatments, healthcare
- legal: Law, regulations, contracts, compliance, intellectual property
- general: Science, history, general knowledge, or anything else

Query: "{query}"

Respond with ONLY the category name (code, medical, legal, or general). No explanation or extra text."""
    
    def __init__(self):
        """Initialize Ollama judge with configuration from environment."""
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
        self.available_domains = ["code", "medical", "legal", "general"]
        
        # Configure Ollama client
        if self.base_url != "http://localhost:11434":
            # Custom URL (e.g., cloud)
            self.client = ollama.Client(host=self.base_url)
        else:
            # Default local
            self.client = ollama.Client()
    
    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify query using Ollama LLM.
        
        Args:
            query: User query to classify
            
        Returns:
            Tuple of (domain, confidence)
            Confidence is 0.95 for valid Ollama responses
        """
        try:
            # Generate classification
            prompt = self.CLASSIFICATION_PROMPT.format(query=query)
            
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.1,  # Low temperature for consistent classification
                    "num_predict": 10,   # Only need 1 word
                }
            )
            
            # Extract domain from response
            response_text = response["message"]["content"].strip().lower()
            
            detected_domain = "general"  # Default fallback
            for domain in self.available_domains:
                if domain in response_text:
                    detected_domain = domain
                    break
            
            # Ollama judge is very confident when it responds
            return detected_domain, 0.95
        
        except Exception as e:
            print(f"Ollama judge error: {e}. Falling back to 'general' domain.")
            return "general", 0.5


# Global instance
_ollama_judge = None


def get_ollama_judge() -> OllamaJudge:
    """Get or create Ollama judge instance."""
    global _ollama_judge
    if _ollama_judge is None:
        _ollama_judge = OllamaJudge()
    return _ollama_judge
