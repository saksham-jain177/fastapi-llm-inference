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

    SYNTHESIS_PROMPT = """Based on the following context, provide a comprehensive answer.

Context:
{context}

Question: {query}

Instructions:
1. Group your findings into logical claims.
2. For every claim, cite the relevant [Source X] marker immediately after the claim.
3. Ensure every sentence is semantically complete. Do not cut off mid-thought.
4. If you have a lot of information, prioritize the most relevant points and finish with a strong summary sentence if you are near the length limit.
5. End your response with a clear concluding punctuation mark."""

    def __init__(self):
        """Initialize Ollama reasoner with configuration."""
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
        self._client = None

    def _get_client(self):
        """Lazy-load Ollama AsyncClient. Import happens here, not at module level."""
        if self._client is None:
            try:
                from ollama import AsyncClient
            except ImportError:
                raise RuntimeError(
                    "Ollama Python module not installed. Install with: pip install ollama. "
                    "Note: The 'ollama serve' process must also be running."
                )
            if self.base_url != "http://localhost:11434":
                self._client = AsyncClient(host=self.base_url)
            else:
                self._client = AsyncClient()
        return self._client

    async def infer(self, prompt: str) -> str:
        """
        Generate a response using Ollama.
        """
        try:
            client = self._get_client()
            response = await client.chat(
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

    async def reason(self, query: str) -> Dict[str, str]:
        """
        Perform multi-step reasoning on a query.
        """
        try:
            client = self._get_client()
            prompt = self.REASONING_PROMPT.format(query=query)

            response = await client.chat(
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

    async def synthesize_with_context(self, query: str, context: str) -> str:
        """
        Synthesize answer from context (for RAG).
        """
        try:
            client = self._get_client()
            prompt = self.SYNTHESIS_PROMPT.format(query=query, context=context)

            response = await client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.2,
                    "num_predict": 1024,
                },
            )

            return response["message"]["content"].strip()

        except Exception as e:
            raise RuntimeError(f"Ollama synthesis failed: {e}") from e
