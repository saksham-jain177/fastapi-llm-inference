"""
Routing Orchestrator.
 Coordinates analysis, retrieval (RAG), and inference strategies.
"""

from app.routing.query_analyzer import get_query_analyzer
from app.routing.reasoner import get_ollama_reasoner
from app.models.adapter_manager import get_adapter_manager
from app.models.quantized import generate_response
from app.rag.retrieval import search_web_context  # Assuming this exists from Phase 2
import time

class Orchestrator:
    def __init__(self):
        self.analyzer = get_query_analyzer()
        self.reasoner = get_ollama_reasoner()
        self.adapter_mgr = get_adapter_manager()
        
    def route_and_execute(self, query: str, feedback_intent: str = None) -> dict:
        """
        Execute full request pipeline:
        1. Analyze Query (Intent Classification)
        2. Route to appropriate handler
        3. Execute (RAG, Reasoning, or Base Inference)
        
        Args:
            query: User's prompt
            feedback_intent: Optional ground truth for metric logging (tp/fp etc)
        """
        # 1. Analysis
        analysis = self.analyzer.analyze(query)
        predicted_intent = analysis["intent"]
        
        # Log accuracy metrics if feedback is provided
        if feedback_intent:
            from app.metrics.prometheus import (
                classification_tp, classification_fp, 
                classification_fn, classification_tn
            )
            # Simple binary check: Match vs Mismatch
            if predicted_intent == feedback_intent:
                classification_tp.labels(intent=predicted_intent).inc()
            else:
                classification_fp.labels(intent=predicted_intent).inc()
                classification_fn.labels(intent=feedback_intent).inc()
        
        response_data = {
            "prompt_received": query,
            "analysis": analysis,
            "timestamp": time.time()
        }
        
        # 2. Routing & Execution
        
        # Path A: External Search (RAG)
        if predicted_intent == "external_search":
            context = search_web_context(query)
            # Synthesize with reasoner (Ollama)
            final_response = self.reasoner.synthesize_with_context(query, context)
            
            response_data.update({
                "mode": "rag-external",
                "response": final_response,
                "context_used": True
            })
            return response_data
            
        # Path B: Complex Reasoning (Chain of Thought)
        elif predicted_intent == "complex_reasoning":
            reasoning_result = self.reasoner.reason(query)
            
            response_data.update({
                "mode": "internal-reasoning",
                "response": reasoning_result["answer"],
                "reasoning_trace": reasoning_result["reasoning"],
                "reasoning_used": True
            })
            return response_data
            
        # Path C: Simple Internal (Adapter/Base)
        else: # simple_internal or fallback
            # Semantic router still useful for domain selection within 'simple' intent
            from app.routing.semantic_router import get_semantic_router
            router = get_semantic_router()
            domain, conf = router.classify(query)
            
            if self.adapter_mgr.has_adapter(domain):
                resp = self.adapter_mgr.generate_with_adapter(domain, query)
                response_data.update({
                    "mode": "adapter",
                    "domain": domain,
                    "response": resp,
                    "adapter_used": True
                })
            else:
                resp = generate_response(query)
                response_data.update({
                    "mode": "base-model",
                    "domain": "general",
                    "response": resp
                })
                
            return response_data

# Global instance
_orchestrator = None

def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
