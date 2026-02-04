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
from app.rag.data_collector import get_data_collector
from app.routing.semantic_router import get_semantic_router
from app.rag.feedback_retriever import get_feedback_retriever
from app.metrics.prometheus import memory_hit_total


class Orchestrator:
    def __init__(self):
        self.analyzer = get_query_analyzer()
        self.reasoner = get_reasoner()  # Factory-provided, interface-only
        self.adapter_mgr = get_adapter_manager()

    def _is_incomplete(self, text: str) -> bool:
        """
        Detects if the response seems truncated based on terminal punctuation.
        """
        if not text:
            return False
            
        text = text.strip()
        
        # Check for valid terminal punctuation
        valid_endings = ('.', '!', '?', '"', "'", '`', '}')
        if text.endswith(valid_endings):
            return False
            
        # Check for trailing stop words that suggest interruption
        trailing_indicators = ('and', 'or', 'but', 'the', 'a', 'an', 'with', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'which', 'that')
        if text.lower().endswith(trailing_indicators):
            return True
            
        # If it doesn't end with valid punctuation, assume incomplete
        return True

    def _clean_citations(self, text: str, citations: list) -> str:
        """
        Enforce citation integrity:
        1. If no citations exist, remove [Source X] markers (hallucinations).
        2. If citations exist but text has no markers, that's fine (frontend handles it).
        """
        import re
        
        # Case 1: No valid citations -> Strip all markers
        if not citations:
            # Matches [Source 1], [Source 10], etc.
            return re.sub(r'\[Source \d+\]', '', text).strip()
            
        return text

    async def route_and_execute(self, query: str, feedback_intent: str = None) -> dict:
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
            if predicted_intent == feedback_intent:
                classification_tp.labels(intent=predicted_intent).inc()
            else:
                classification_fp.labels(intent=predicted_intent).inc()
                classification_fn.labels(intent=feedback_intent).inc()
        
        response_data = {
            "prompt_received": query,
            "analysis": analysis,
            "timestamp": time.time(),
            "refused": False,
            "confidence": 0.0,
            "source": "model",
            "intent": predicted_intent
        }
        
        # Redis Hot Cache Lookup
        collector = get_data_collector()
        if hasattr(collector, 'get_cached_response'):
            cached_resp = await collector.get_cached_response(query)
            if cached_resp:
                print(f"  ⚡ Redis Cache Hit")
                response_data.update({
                    "mode": "redis_cache",
                    "response": cached_resp,
                    "cache_hit": True,
                    "confidence": 1.0,
                    "source": "redis"
                })
                return response_data

        # 3. Routing & Execution
        context = ""
        final_response = ""
        log_intent = predicted_intent

        # Path A: External Search (RAG)
        if predicted_intent == "external_search":
            try:
                # 1. Memory lookup (Short-circuit if we have high-confidence grounded answer)
                memory_retriever = get_feedback_retriever()
                if memory_retriever:
                    # search_similar is SYNC
                    memory_hits = memory_retriever.search_similar(query, top_k=1)
                    if memory_hits:
                        match = memory_hits[0]
                        eff_conf = match["confidence"]
                        if eff_conf >= 0.9:
                            print(f"  🧠 Memory hit! Reusing high-confidence answer")
                            memory_hit_total.inc()
                            await collector.cache_response(query, match["response"])
                            response_data.update({
                                "mode": "memory",
                                "response": match["response"],
                                "memory_used": True,
                                "similarity": match["similarity"],
                                "confidence": eff_conf,
                                "source": "memory"
                            })
                            await collector.log_interaction(query, "memory-hit", match["response"], "external_search", confidence=eff_conf, source="memory")
                            return response_data
            except Exception as e:
                print(f"Memory lookup failed: {e}")

            final_response, context, citations, log_intent = await self._execute_external_rag(query, collector, response_data)

        # Path B: Complex Reasoning
        elif predicted_intent == "complex_reasoning":
            reasoning_result = await self.reasoner.reason(query)
            final_response = reasoning_result["answer"]
            response_data.update({
                "mode": "internal-reasoning",
                "response": final_response,
                "reasoning_trace": reasoning_result["reasoning"],
                "reasoning_used": True,
                "confidence": 1.0
            })

        # Path C: Simple Internal
        else:
            router = get_semantic_router()
            domain, conf = router.classify(query)

            # FORCE RAG/REFUSAL if domain is unknown (below semantic threshold)
            # This is the "Ambiguity is terminal" rule
            if domain == "unknown":
                print(f"[Orchestrator] Domain unknown (semantic_conf: {conf:.3f}), blocking model path")
                final_response, context, citations, log_intent = await self._execute_external_rag(query, collector, response_data)
            elif self.adapter_mgr.has_adapter(domain):
                # Sync call for model inference (as requested)
                resp = self.adapter_mgr.generate_with_adapter(domain, query)
                final_response = resp
                response_data.update({
                    "mode": "adapter",
                    "domain": domain,
                    "response": resp,
                    "adapter_used": True,
                    "confidence": conf
                })
            else:
                # Path C: Epistemic Gating
                from app.models.confidence import get_confidence_estimator
                from app.models.calibration import get_confidence_threshold
                from app.routing.retrieval_gate import get_retrieval_gate
                from app.constants import CANONICAL_REFUSAL
                from app.metrics.prometheus import gate_decision_total, hallucination_counter, refusal_counter, inference_latency

                estimator = get_confidence_estimator()
                
                # Wrapper to use with estimator
                async def generate_wrapper(p, temperature=0.3, max_new_tokens=256):
                    # Logic to use base model via adapter_mgr (which handles lazy loading)
                    return self.adapter_mgr.generate_with_adapter("base", p, temperature=temperature, max_new_tokens=max_new_tokens)

                print(f"[Orchestrator] Entering epistemic gate for query: '{query}'")
                draft_response, epistemic_conf = await estimator.estimate_confidence(query, generate_wrapper)
                
                gate_threshold = get_confidence_threshold("simple_internal")
                
                # RULE: Ambiguity is terminal.
                # Reliability = Semantic AND Epistemic
                is_reliable = (conf >= 0.35) and (epistemic_conf >= gate_threshold)

                if is_reliable:
                    print(f"  → Gating PASSED (semantic: {conf:.2f}, epistemic: {epistemic_conf:.2f})")
                    final_response = draft_response
                    response_data.update({
                        "mode": "model",
                        "response": final_response,
                        "confidence": epistemic_conf,  # We report epistemic here as it's the more conservative one
                        "semantic_confidence": conf
                    })
                    gate_decision_total.labels(decision="allow", reason="high_confidence").inc()
                else:
                    print(f"  → Gating FAILED (semantic: {conf:.2f}, epistemic: {epistemic_conf:.2f})")
                    # Check novelty to decide between RAG or Refusal
                    retrieval_gate = get_retrieval_gate()
                    should_search = retrieval_gate.should_trigger_retrieval(query)
                    
                    if should_search:
                        print("    → Novelty high, falling back to RAG")
                        final_response, context, citations, log_intent = await self._execute_external_rag(query, collector, response_data)
                        gate_decision_total.labels(decision="fallback_rag", reason="low_confidence_novel").inc()
                    else:
                        print("    → Novelty low but unsure, refusing to guess")
                        final_response = CANONICAL_REFUSAL
                        response_data.update({
                            "mode": "refused",
                            "response": final_response,
                            "refused": True,
                            "source": "refused",
                            "confidence": epistemic_conf
                        })
                        gate_decision_total.labels(decision="refuse", reason="low_confidence_stale").inc()
                        refusal_counter.inc()
                        log_intent = "refused"

        # Final Centralized Awaited Side Effect
        await collector.log_interaction(
            query=query,
            context=context or "none",
            response=final_response,
            intent=log_intent,
            confidence=response_data.get("confidence", 0.0),
            source=response_data.get("source", "model")
        )
        
        return response_data

    async def _execute_external_rag(self, query: str, collector, response_data: dict):
        """Helper to execute external RAG (Tavily + Synthesis)."""
        import os
        
        context = "none"
        citations = []
        log_intent = "external_search"
        final_response = ""

        if not os.getenv("TAVILY_API_KEY"):
            final_response = "I cannot search for external information at this time. This capability is currently unavailable."
            response_data.update({"mode": "refused", "response": final_response, "refused": True, "source": "refused"})
        else:
            try:
                # Sync call for IO
                context, citations = search_web_context(query)
                final_response = await self.reasoner.synthesize_with_context(query, context)
                
                if self._is_incomplete(final_response):
                    final_response += "\n\n(This answer may be incomplete. You can ask a follow-up.)"
                final_response = self._clean_citations(final_response, citations)
                
                await collector.cache_response(query, final_response)
                response_data.update({
                    "mode": "rag-external",
                    "response": final_response,
                    "context_used": True,
                    "confidence": 1.0,
                    "source": "rag",
                    "citations": citations
                })
                log_intent = "rag-external"
            except Exception as e:
                print(f"RAG failed: {e}")
                final_response = "External search failed or timed out."
                response_data.update({"mode": "refused", "response": final_response, "refused": True, "source": "refused"})
        
        return final_response, context, citations, log_intent

# Global instance
_orchestrator = None

def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
