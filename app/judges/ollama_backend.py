"""
Ollama-based Judge implementation.
Lazy-loads ollama at runtime to prevent import-time hardware requirement.
"""

import os
from typing import Tuple
from app.judges.base import Judge


class OllamaJudge(Judge):
    """
    Uses Ollama LLM to classify queries.
    Imports ollama only inside methods.
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
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
        self.available_domains = ["code", "medical", "legal", "general"]
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
            except ImportError:
                raise RuntimeError(
                    "Ollama Python module not installed. Install with: pip install ollama. "
                    "Note: The 'ollama serve' process must also be running."
                )
            if self.base_url != "http://localhost:11434":
                self._client = ollama.Client(host=self.base_url)
            else:
                self._client = ollama.Client()
        return self._client

    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify query using Ollama LLM.
        """
        try:
            client = self._get_client()
            prompt = self.CLASSIFICATION_PROMPT.format(query=query)
            
            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.1,
                    "num_predict": 10,
                }
            )
            
            response_text = response["message"]["content"].strip().lower()
            
            detected_domain = "general"
            for domain in self.available_domains:
                if domain in response_text:
                    detected_domain = domain
                    break
            
            return detected_domain, 0.95
        
        except Exception as e:
            # Fallback within the backend if provider fails
            return "general", 0.5
