"""
Deterministic Reasoner for hardware-independent inference.
Used in CI, local logic testing, and as fallback when hardware is unavailable.

No external imports. Zero randomness. Stable across runs.
"""

import hashlib
from typing import Dict

from app.reasoners.base import Reasoner


class DeterministicReasoner(Reasoner):
    """
    Hardware-independent backend for deterministic inference.
    Produces stable, predictable outputs based on prompt hashing.
    """

    # Static response templates for common patterns
    TEMPLATES = {
        "capital": "The capital of {subject} is a major city known for its historical significance.",
        "definition": "{subject} is a concept or entity with specific characteristics.",
        "how": "To accomplish this, you would follow a structured approach involving multiple steps.",
        "why": "This occurs due to a combination of factors that interact in complex ways.",
        "default": "Based on the query, here is a structured response addressing the key points.",
    }

    async def infer(self, prompt: str) -> str:
        """
        Generate a deterministic response based on prompt hashing.
        """
        prompt_lower = prompt.lower()

        # Pattern matching for template selection
        if "capital" in prompt_lower:
            template = self.TEMPLATES["capital"]
            # Extract subject heuristically
            subject = self._extract_subject(prompt)
            return template.format(subject=subject)

        if any(word in prompt_lower for word in ["what is", "define", "explain"]):
            template = self.TEMPLATES["definition"]
            subject = self._extract_subject(prompt)
            return template.format(subject=subject)

        if prompt_lower.startswith("how"):
            return self.TEMPLATES["how"]

        if prompt_lower.startswith("why"):
            return self.TEMPLATES["why"]

        # Fallback: hash-based response with stable suffix
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        return f"{self.TEMPLATES['default']} [Response ID: {prompt_hash}]"

    async def reason(self, query: str) -> Dict[str, str]:
        """
        Perform deterministic multi-step reasoning.
        """
        prompt_hash = hashlib.md5(query.encode()).hexdigest()[:8]

        reasoning = (
            f"1. Understanding: The query asks about '{query[:50]}...'\n"
            f"2. Analysis: Key components identified.\n"
            f"3. Reasoning: Applying structured logic.\n"
        )

        answer = f"Based on deterministic analysis, the answer addresses the core question. [ID: {prompt_hash}]"

        return {
            "reasoning": reasoning,
            "answer": answer,
            "full_response": f"{reasoning}\n\nAnswer: {answer}",
        }

    async def synthesize_with_context(self, query: str, context: str) -> str:
        """
        Synthesize answer from context deterministically.
        """
        context_hash = hashlib.md5(context.encode()).hexdigest()[:6]
        query_hash = hashlib.md5(query.encode()).hexdigest()[:6]

        return (
            f"Based on the provided context, the answer to '{query[:40]}...' "
            f"incorporates relevant information. [Context: {context_hash}, Query: {query_hash}]"
        )

    def _extract_subject(self, prompt: str) -> str:
        """Extract a subject from the prompt for template filling."""
        # Simple heuristic: take the last significant word
        words = prompt.replace("?", "").replace(".", "").split()
        if len(words) >= 2:
            return words[-1]
        return "the subject"
