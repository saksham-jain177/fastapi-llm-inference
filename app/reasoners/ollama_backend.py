"""
Ollama Reasoner backend.
Implements the Reasoner interface with lazy-loaded Ollama imports.

CRITICAL: `import ollama` is ONLY inside execution scope, never at top level.
"""

import os
from typing import Dict

from app.reasoners.base import Reasoner


class OllamaReasoner(Reasoner):
    """
    Concrete Ollama implementation of the Reasoner interface.
    All Ollama imports are deferred to execution time.
    """

    REASONING_PROMPT = """You are a helpful AI assistant that thinks step-by-step.

For the following question, break down your reasoning into clear steps:
1. Understanding: Rephrase the question in your own words
2. Analysis: Identify key components and what's being asked
3. Reasoning: Think through the solution step-by-step
4. Answer: Provide the final answer

Question: {query}

Think carefully and show your reasoning:"""

    SYNTHESIS_PROMPT = """Based on the following information, provide a clear and concise answer.

Context:
{context}

Question: {query}

Synthesize the information above and provide a comprehensive answer.
Cite your sources using [Source X] format where appropriate."""

    def __init__(self):
        """Initialize Ollama reasoner with configuration."""
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
        self._client = None

    def _get_client(self):
        """Lazy-load Ollama client. Import happens here, not at module level."""
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

    def infer(self, prompt: str) -> str:
        """
        Generate a response using Ollama.

        Args:
            prompt: User's input prompt

        Returns:
            Generated text response
        """
        try:
            client = self._get_client()
            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.3,
                    "num_predict": 256,
                },
            )
            return response["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Ollama inference failed: {e}") from e

    def reason(self, query: str) -> Dict[str, str]:
        """
        Perform multi-step reasoning on a query.

        Args:
            query: Complex query requiring reasoning

        Returns:
            Dictionary with 'reasoning' (steps) and 'answer' (final result)
        """
        try:
            client = self._get_client()
            prompt = self.REASONING_PROMPT.format(query=query)

            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.3,
                    "num_predict": 512,
                },
            )

            full_response = response["message"]["content"]

            # Extract structured parts
            if "Answer:" in full_response:
                parts = full_response.split("Answer:", 1)
                reasoning = parts[0].strip()
                answer = parts[1].strip()
            else:
                paragraphs = full_response.split("\n\n")
                if len(paragraphs) > 1:
                    reasoning = "\n\n".join(paragraphs[:-1])
                    answer = paragraphs[-1]
                else:
                    reasoning = full_response
                    answer = full_response

            return {
                "reasoning": reasoning,
                "answer": answer,
                "full_response": full_response,
            }

        except Exception as e:
            raise RuntimeError(f"Ollama reasoning failed: {e}") from e

    def synthesize_with_context(self, query: str, context: str) -> str:
        """
        Synthesize answer from context (for RAG).

        Args:
            query: User query
            context: Retrieved context from Tavily/RAG

        Returns:
            Synthesized answer
        """
        try:
            client = self._get_client()
            prompt = self.SYNTHESIS_PROMPT.format(query=query, context=context)

            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.2,
                    "num_predict": 300,
                },
            )

            return response["message"]["content"].strip()

        except Exception as e:
            raise RuntimeError(f"Ollama synthesis failed: {e}") from e
