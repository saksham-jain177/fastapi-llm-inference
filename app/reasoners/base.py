"""
Abstract Reasoner interface.
No third-party imports. No logic. No side effects.
"""

from abc import ABC, abstractmethod
from typing import Dict


class Reasoner(ABC):
    """
    Abstract interface for inference backends.
    All concrete implementations must inherit from this class.
    """

    @abstractmethod
    def infer(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: User's input prompt

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def reason(self, query: str) -> Dict[str, str]:
        """
        Perform multi-step reasoning on a query.

        Args:
            query: Complex query requiring reasoning

        Returns:
            Dictionary with 'reasoning' (steps) and 'answer' (final result)
        """
        pass

    def synthesize_with_context(self, query: str, context: str) -> str:
        """
        Synthesize answer from context (for RAG).
        Default implementation delegates to infer().

        Args:
            query: User query
            context: Retrieved context from RAG

        Returns:
            Synthesized answer
        """
        combined_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a concise answer based on the context."
        return self.infer(combined_prompt)
