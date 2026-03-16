"""
Routing Orchestrator.
 Coordinates analysis, retrieval (RAG), and inference strategies.
 
 Depends only on abstract Reasoner interface. Backend selection is deferred to runtime via factory.
"""

from app.reasoners.factory import get_reasoner
from app.models.adapter_manager import get_adapter_manager
from app.rag.retrieval import search_web_context
import asyncio
import time
from app.rag.data_collector import get_data_collector
from app.routing.semantic_router import get_semantic_router
from app.rag.context_manager import get_context_packer
from app.routing.information_gate import is_informative


from app.routing.response_utils import is_incomplete, clean_citations

from app.observability.telemetry import get_telemetry_logger

class Orchestrator:
    def __init__(self):
        self.reasoner = get_reasoner()  # Factory-provided, interface-only
        self.adapter_mgr = get_adapter_manager()

    async def route_and_execute(self, query: str, feedback_intent: str = None, headers: dict = None) -> dict:
        """
        Execute full request pipeline using Truth-First Inference.
        
        Path:
        0. Information Density Check (Early refusal)
        1. Semantic Classification (Domain)
        2. Adapter Check (Specialized Knowledge)
        3. Knowledge Gate (General Knowledge)
           -> RAG (if unknown/uncertain) 
           -> Model (only if strictly confident)
        """
        # Telemetry: Load Test Audit
        if headers and headers.get("x-load-test"):
            telemetry = get_telemetry_logger()
            await telemetry.log_event("load_test_request", {
                "query": query,
                "timestamp": time.time(),
                "headers": headers
            })

        # 0. Information Density Check
        if not is_informative(query):
            print(f"  🛑 Low information query refused: '{query}'")
            refusal_response = "Please provide a more specific, information-seeking question."
            
            response_data = {
                "prompt_received": query,
                "timestamp": time.time(),
                "refused": True,
                "confidence": 0.0,
                "source": "refused",
                "response": refusal_response,
                "mode": "refused"
            }
            
            # Log refusal
            collector = get_data_collector()
            await collector.log_interaction(
                query=query,
                context="none",
                response=refusal_response,
                intent="refused",
                confidence=0.0,
                source="refused"
            )
            return response_data

        # 1. Semantic Classification
        router = get_semantic_router()
        domain, semantic_conf = router.classify(query)
        
        response_data = {
            "prompt_received": query,
            "timestamp": time.time(),
            "refused": False,
            "confidence": 0.0,
            "source": "model",
            "intent": domain
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

        context = ""
        final_response = ""
        log_intent = domain

        # Path A: Specialized Adapter (Strictly matched domain)
        if self.adapter_mgr.has_adapter(domain):
            print(f"[Orchestrator] Using adapter for domain: {domain}")
            # Sync call for model inference
            resp = self.adapter_mgr.generate_with_adapter(domain, query)
            final_response = resp
            response_data.update({
                "mode": "adapter",
                "domain": domain,
                "response": resp,
                "adapter_used": True,
                "confidence": semantic_conf
            })
            
        # Path B: Knowledge Gate (General/Unknown)
        else:
            from app.models.confidence import get_confidence_estimator
            from app.routing.knowledge_gate import get_knowledge_gate
            from app.constants import CANONICAL_REFUSAL
            from app.metrics.prometheus import gate_decision_total, refusal_counter

            gate = get_knowledge_gate()
            
            # CRITICAL INSIGHT:
            # For general queries, we have NO EVIDENCE until we search.
            # The model can generate a draft, but that's not evidence—it's speculation.
            # 
            # Evidence sources:
            # - Adapters (domain-specific knowledge) ← handled in Path A
            # - RAG retrieval (external grounding) ← not yet available
            # - Internal KB/docs ← not implemented
            #
            # Therefore: has_evidence = False for all general queries
            
            has_evidence = False
            
            # DECIDE via Knowledge Gate (Evidence-First)
            # We pass epistemic_confidence=None because we haven't generated yet
            decision = gate.decide(
                semantic_score=semantic_conf,
                has_evidence=has_evidence,
                epistemic_confidence=None  # Not relevant without evidence
            )
            
            print(f"  → KnowledgeGate Decision: {decision.upper()} (semantic: {semantic_conf:.2f}, has_evidence: {has_evidence})")
            
            if decision == "rag":
                # Search for evidence
                print("    → No evidence available, searching...")
                final_response, context, citations, log_intent = await self._execute_external_rag(query, collector, response_data)
                gate_decision_total.labels(decision="fallback_rag", reason="no_evidence").inc()
                
            elif decision == "refuse":
                print("    → Explicit refusal (no safe path)")
                final_response = CANONICAL_REFUSAL
                response_data.update({
                    "mode": "refused",
                    "response": final_response,
                    "refused": True,
                    "source": "refused",
                    "confidence": 0.0
                })
                gate_decision_total.labels(decision="refuse", reason="no_evidence").inc()
                refusal_counter.inc()
                log_intent = "refused"
                
            elif decision == "model":
                # This path should NEVER be reached for general queries without evidence
                # But if it somehow is (e.g., future KB integration), we need epistemic check
                estimator = get_confidence_estimator()
                
                async def generate_wrapper(p, temperature=0.3, max_new_tokens=256):
                    return self.adapter_mgr.generate_with_adapter("base", p, temperature=temperature, max_new_tokens=max_new_tokens)
                
                print(f"[Orchestrator] Evidence-backed path - checking epistemic confidence")
                draft_response, epistemic_conf = await estimator.estimate_confidence(query, generate_wrapper)
                
                # Re-check with epistemic confidence
                final_decision = gate.decide(
                    semantic_score=semantic_conf,
                    has_evidence=True,  # Assuming we got here via evidence
                    epistemic_confidence=epistemic_conf
                )
                
                if final_decision == "model":
                    final_response = draft_response
                    response_data.update({
                        "mode": "model",
                        "response": final_response,
                        "confidence": epistemic_conf,
                        "semantic_confidence": semantic_conf
                    })
                    gate_decision_total.labels(decision="allow", reason="evidence_backed").inc()
                else:
                    # Epistemic check failed
                    print("    → Epistemic confidence too low, refusing")
                    final_response = CANONICAL_REFUSAL
                    response_data.update({
                        "mode": "refused",
                        "response": final_response,
                        "refused": True,
                        "source": "refused",
                        "confidence": epistemic_conf
                    })
                    gate_decision_total.labels(decision="refuse", reason="low_epistemic").inc()
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

        try:
            # 1. Search (Fetches more base results now)
            status, raw_results = await search_web_context(query)
            
            # 2. Rank and Pack
            packer = get_context_packer()
            context, citations = packer.pack(query, raw_results)
            
            # 3. Synthesize (Async)
            # Wrap in timeout for robustness
            final_response = await asyncio.wait_for(
                self.reasoner.synthesize_with_context(query, context),
                timeout=30.0
            )
            
            if is_incomplete(final_response):
                final_response += "\n\n(This answer may be incomplete. You can ask a follow-up.)"
            final_response = clean_citations(final_response, citations)
            
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
        except asyncio.TimeoutError:
            print(f"RAG timed out for query: {query}")
            final_response = "External search or synthesis timed out. Please try a simpler query."
            response_data.update({"mode": "refused", "response": final_response, "refused": True, "source": "refused"})
        except Exception as e:
            print(f"RAG failed: {e}")
            final_response = "External search failed."
            response_data.update({"mode": "refused", "response": final_response, "refused": True, "source": "refused"})
        
        return final_response, context, citations, log_intent

# Global instance and lock
import threading
_orchestrator_lock = threading.Lock()
_orchestrator = None

def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        with _orchestrator_lock:
            if _orchestrator is None:
                _orchestrator = Orchestrator()
    return _orchestrator
