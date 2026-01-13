"""
Deterministic Judge for hardware-independent CI and logic testing.
Zero randomness, no hardware required.
"""

import hashlib
from typing import Tuple
from app.judges.base import Judge


class DeterministicJudge(Judge):
    """
    Produces stable, predictable classifications based on prompt contents.
    """

    DOMAINS = ["code", "medical", "legal", "general"]

    def classify(self, query: str) -> Tuple[str, float]:
        """
        Simple keyword-based deterministic classification.
        """
        q_lower = query.lower()
        
        # Simple priority logic
        if any(w in q_lower for w in ["def ", "class ", "import ", "{", "code"]):
            return "code", 0.95
        if any(w in q_lower for w in ["disease", "diagnosis", "medical", "patient"]):
            return "medical", 0.95
        if any(w in q_lower for w in ["law", "contract", "legal", "statute"]):
            return "legal", 0.95
            
        return "general", 0.90
