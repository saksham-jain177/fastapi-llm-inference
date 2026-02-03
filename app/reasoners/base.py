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
    async def infer(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.
        """
        pass

    @abstractmethod
    async def reason(self, query: str) -> Dict[str, str]:
        """
        Perform multi-step reasoning on a query.
        """
        pass

    async def synthesize_with_context(self, query: str, context: str) -> str:
        """
        Synthesize answer from context (for RAG).
        """
        combined_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a concise answer based on the context."
        return await self.infer(combined_prompt)
