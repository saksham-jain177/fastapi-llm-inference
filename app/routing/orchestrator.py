"""
Routing Orchestrator.
 Coordinates analysis, retrieval (RAG), and inference strategies.
 
 Depends only on abstract Reasoner interface. Backend selection is deferred to runtime via factory.
"""

from app.routing.query_analyzer import get_query_analyzer
from app.reasoners.factory import get_reasoner
from app.models.adapter_manager import get_adapter_manager
from app.rag.retrieval import search_web_context
import asyncio
import time


class Orchestrator:
    def __init__(self):
        self.analyzer = get_query_analyzer()
        self.reasoner = get_reasoner()  # Factory-provided, interface-only
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
            
            # Log for continuous learning (A+B=AB)
            from app.rag.data_collector import get_data_collector
            collector = get_data_collector()
            # Fire-and-forget logging to avoid blocking sync path
            asyncio.create_task(collector.log_interaction(
                query=query,
                context=context,
                response=final_response,
                intent="rag-external"
            ))
            
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
                # Path C: Epistemic Gating (Confidence-Based Abstention)
                from app.models.confidence import get_confidence_estimator
                from app.models.calibration import get_confidence_threshold
                from app.routing.retrieval_gate import get_retrieval_gate
                from app.constants import CANONICAL_REFUSAL
                from app.metrics.prometheus import gate_decision_total, hallucination_counter, refusal_counter, inference_latency
                
                start_time = time.time()
                print(f"\n🧠 Epistemic Gating for: '{query}'")
                
                # Use reasoner interface for confidence estimation
                def generate_fn(q: str, temperature: float = 0.1, max_new_tokens: int = 50) -> str:
                    """Wrapper for confidence estimation using the abstract reasoner."""
                    return self.reasoner.infer(q)
                
                # Estimate confidence (includes perturbation check)
                estimator = get_confidence_estimator()
                draft_response, confidence = estimator.estimate_confidence(query, generate_fn)

                
                # Get auto-calibrated threshold
                threshold = get_confidence_threshold()
                
                print(f"  Confidence: {confidence:.3f} (threshold: {threshold:.3f})")
                
                # Confidence gate
                if confidence >= threshold:
                    # HIGH CONFIDENCE - Answer directly
                    print(f"  ✅ Confident answer")
                    
                    # Update knowledge centroid for future retrieval gating
                    get_retrieval_gate().update_centroid(query)
                    
                    gate_decision_total.labels(type="answer").inc()
                    inference_latency.labels(mode="confident").observe(time.time() - start_time)
                    
                    response_data.update({
                        "mode": "internal-confident",
                        "response": draft_response,
                        "confidence": confidence
                    })
                    return response_data
                
                # LOW CONFIDENCE - Check retrieval eligibility
                print(f"  ⚠️ Low confidence - checking retrieval eligibility")
                
                if get_retrieval_gate().should_retrieve(query):
                    # Novel query - trigger retrieval
                    print(f"  🔍 Novel query detected - triggering retrieval")
                    gate_decision_total.labels(type="search").inc()
                    
                    context = search_web_context(query)
                    
                    # Synthesize with reasoner (Ollama)
                    synthesized = self.reasoner.synthesize_with_context(query, context)
                    
                    inference_latency.labels(mode="search").observe(time.time() - start_time)
                    
                    response_data.update({
                        "mode": "rag",
                        "response": synthesized,
                        "context": context,
                        "confidence": confidence,
                        "retrieved": True
                    })
                    return response_data
                
                # LOW NOVELTY + LOW CONFIDENCE -> FORCED ABSTENTION
                print(f"  ⛔ Forced abstention (hallucination risk)")
                gate_decision_total.labels(type="refuse").inc()
                hallucination_counter.inc()
                refusal_counter.inc()
                inference_latency.labels(mode="fallback").observe(time.time() - start_time)
                
                # Update knowledge centroid anyway to mark we've "seen" the ignorance
                get_retrieval_gate().update_centroid(query)
                
                response_data.update({
                    "mode": "abstained",
                    "response": CANONICAL_REFUSAL,
                    "confidence": confidence,
                    "abstained": True
                })
                return response_data

# Global instance
_orchestrator = None

def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
