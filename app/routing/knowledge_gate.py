"""
Knowledge Gate.
Single authority for "may the model answer directly?"

Truth-First Principle:
The LLM is a language synthesizer, NOT a knowledge oracle.
It may only speak when backed by evidence.
"""

from typing import Literal, Optional

class KnowledgeGate:
    """
    Evidence-first gating logic.
    
    Order of checks:
    1. Semantic certainty (do we recognize this domain?)
    2. Evidence availability (do we have grounding?)
    3. Epistemic confidence (is the model internally consistent?)
    """
    
    def __init__(self, semantic_threshold: float = 0.35, epistemic_threshold: float = 0.75):
        self.semantic_threshold = semantic_threshold
        self.epistemic_threshold = epistemic_threshold
    
    def decide(
        self, 
        semantic_score: float, 
        has_evidence: bool,
        epistemic_confidence: Optional[float] = None
    ) -> Literal["rag", "model", "refuse"]:
        """
        Decide routing based on EVIDENCE, not confidence.
        
        Rules (in order):
        1. If semantic_score < threshold → RAG (unknown domain)
        2. If has_evidence == False → RAG (no grounding)
        3. If epistemic_confidence < threshold → RAG or REFUSE
        4. Otherwise → MODEL
        
        Args:
            semantic_score: How well the query matches known domains (0-1)
            has_evidence: Whether we have external grounding (KB/docs/search)
            epistemic_confidence: Model's internal consistency (optional)
        
        Returns:
            "rag": Search for evidence
            "model": Answer directly (evidence-backed)
            "refuse": Cannot answer safely
        """
        
        # Rule 1: Unknown domain (semantic uncertainty)
        if semantic_score < self.semantic_threshold:
            return "rag"
        
        # Rule 2: No evidence (CRITICAL - prevents hallucination)
        # This is the core truth-first check:
        # "If there is no evidence, the model may NOT answer factual queries"
        if not has_evidence:
            return "rag"
        
        # Rule 3: Epistemic uncertainty (model isn't sure)
        # Only checked AFTER evidence is confirmed
        if epistemic_confidence is not None:
            if epistemic_confidence < self.epistemic_threshold:
                return "refuse"
        
        # Rule 4: All checks passed - model may answer
        return "model"


# Global instance
_knowledge_gate = None

def get_knowledge_gate() -> KnowledgeGate:
    global _knowledge_gate
    if _knowledge_gate is None:
        from app.models.calibration import get_confidence_threshold
        thresh = get_confidence_threshold()
        _knowledge_gate = KnowledgeGate(
            semantic_threshold=0.35,
            epistemic_threshold=thresh
        )
    return _knowledge_gate
