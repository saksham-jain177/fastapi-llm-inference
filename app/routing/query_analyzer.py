"""
Query Analyzer Module.
Implements feature extraction, rule-based classification, and LLM adjudication.
Aligns with ORCAS-I taxonomy for intent classification.
"""

import re
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time

# Feature Extraction Constants
DOMAIN_MARKERS = {
    "code": {"def ", "class ", "import ", "function", "api", "bug", "error", "exception", "compiler", "runtime", "python", "javascript"},
    "math": {"equation", "calc", "theorem", "proof", "integral", "derivative", "matrix", "algebra"},
    "medical": {"symptom", "disease", "treatment", "diagnosis", "patient", "clinical", "drug", "therapy"},
    "legal": {"law", "regulation", "contract", "Statute", "compliance", "copyright", "liability", "court"}
}

TASK_VERBS = {
    "analyze", "compare", "contrast", "evaluate", "synthesize", "explain", "describe", 
    "generate", "optimize", "refactor", "debug", "summarize", "list", "identify"
}

REASONING_INDICATORS = {
    "because", "therefore", "consequently", "due to", "implies", "leads to", "cause", "effect", 
    "justify", "verify", "step-by-step", "chain of thought", "reasoning"
}

REALTIME_INDICATORS = {
    "news", "latest", "current", "today", "yesterday", "recent", "weather", "stock", "price", 
    "event", "schedule", "release date", "traffic", "2024", "2025"
}


@dataclass
class QueryFeatures:
    length: int
    task_verbs: List[str]
    domain_markers: List[str]
    reasoning_indicators: List[str]
    realtime_indicators: List[str]
    is_complex_structure: bool
    has_code_block: bool


class QueryAnalyzer:
    """
    Analyzes queries to determine intent and complexity using a hybrid approach:
    1. Deterministic Rule Engine (Fast)
    2. LLM Adjudicator (Slow, used for ambiguity)
    """
    
    def __init__(self):
        from app.routing.llm_judge import get_ollama_judge
        self.llm_judge = get_ollama_judge()

    def extract_features(self, query: str) -> QueryFeatures:
        """Extract linguistic and structural features from query."""
        q_lower = query.lower()
        tokens = re.findall(r'\w+', q_lower)
        
        return QueryFeatures(
            length=len(tokens),
            task_verbs=[v for v in TASK_VERBS if v in q_lower],
            domain_markers=[d for d, markers in DOMAIN_MARKERS.items() for m in markers if m in q_lower],
            reasoning_indicators=[r for r in REASONING_INDICATORS if r in q_lower],
            realtime_indicators=[r for r in REALTIME_INDICATORS if r in q_lower],
            is_complex_structure="," in query or ";" in query or " and " in q_lower,
            has_code_block="```" in query or "{" in query
        )

    def _rule_engine_classify(self, query: str, features: QueryFeatures) -> Tuple[str, float]:
        """
        Apply deterministic rules to classify intent/complexity.
        Returns (classification, confidence).
        
        Classifications:
        - simple_internal: Factual/Lookup, handled by base model/adapter
        - complex_reasoning: Needs CoT/Reasoning module
        - external_search: Needs RAG/Tavily
        - ambiguous: Needs LLM adjudication
        """
        # Rule 1: Explicit Real-time intent -> External Search
        if features.realtime_indicators:
            return "external_search", 0.9
            
        # Rule 2: Code generation/debugging -> Simple Internal (Code Adapter)
        if "code" in features.domain_markers or features.has_code_block:
            # If asking for explanation of code -> Complex
            if any(v in ["explain", "analyze", "describe"] for v in features.task_verbs):
                return "complex_reasoning", 0.85
            return "simple_internal", 0.9
            
        # Rule 3: Complex Reasoning triggers
        # Multiple task verbs OR reasoning indicators OR complex structure + length
        complexity_score = (
            len(features.task_verbs) * 2 + 
            len(features.reasoning_indicators) * 3 + 
            (1 if features.is_complex_structure else 0)
        )
        
        if complexity_score >= 3:
            return "complex_reasoning", 0.8
            
        # Rule 4: Simple Factual
        if features.length < 15 and complexity_score == 0:
            return "simple_internal", 0.8
            
        return "ambiguous", 0.0

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Main analysis pipeline.
        
        Returns dictionary with:
        - intent: simple_internal | complex_reasoning | external_search
        - confidence: float
        - source: rule_engine | llm_adjudicator
        - features: dict
        """
        start_time = time.time()
        features = self.extract_features(query)
        
        # Step 1: Rule Engine
        classification, confidence = self._rule_engine_classify(query, features)
        
        result = {
            "query": query,
            "features": features.__dict__,
            "intent": classification,
            "confidence": confidence,
            "source": "rule_engine",
            "latency": 0
        }
        
        # Step 2: LLM Adjudication (if ambiguous)
        if classification == "ambiguous" or confidence < 0.6:
            # Fallback to LLM Judge (using existing Ollama client but simpler prompt)
            # We map LLM domains to our intents for now 
            # (Note: In a full impl, we'd use a specific intent prompt)
            domain, judge_conf = self.llm_judge.classify(query)
            
            # Map domain/judge result to intent
            # This is a simplification; ideally LLM judge would output intents directly
            if domain == "general":
                 # Assume general queries might need search if not simple
                 result["intent"] = "complex_reasoning" # Safe default
            else:
                 result["intent"] = "simple_internal" # Valid domain adapter found
                 
            result["confidence"] = judge_conf
            result["source"] = "llm_adjudicator"
            
        result["latency"] = time.time() - start_time
        return result


# Global instance
_analyzer = None

def get_query_analyzer() -> QueryAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = QueryAnalyzer()
    return _analyzer
