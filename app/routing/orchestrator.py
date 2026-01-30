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
        import asyncio
        from functools import partial
        
        # Helper for running blocking IO/Compute in threadpool
        async def run_sync(func, *args, **kwargs):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, partial(func, *args, **kwargs))

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
        
        # 2a. Redis Hot Cache Lookup
        from app.rag.data_collector import get_data_collector
        collector = get_data_collector()
        if hasattr(collector, 'get_cached_response'):
            cached_resp = await collector.get_cached_response(query)
            if cached_resp:
                print(f"  ⚡ Redis Cache Hit")
                response_data.update({
                    "mode": "redis_cache",
                    "response": cached_resp,
                    "cache_hit": True,
                    "confidence": 1.0,  # Cached answers are high confidence by definition
                    "source": "redis",
                    "intent": predicted_intent,
                    "refused": False
                })
                return response_data

        # 2b. Memory Lookup (Mongo/Chroma)
        from app.rag.feedback_retriever import get_feedback_retriever
        from app.metrics.prometheus import memory_hit_total, memory_miss_total
        
        try:
            memory = get_feedback_retriever()
            # Threshold chosen for high confidence reuse (Configurable)
            import os
            memory_threshold = float(os.getenv("MEMORY_ACCEPT_THRESHOLD", "0.85"))
            
            # search_similar now returns matches sorted by 'confidence'
            matches = memory.search_similar(query, top_k=1, min_similarity=0.80)
            
            if matches:
                match = matches[0]
                eff_conf = match.get("confidence", 0.0)
                
                if eff_conf >= memory_threshold:
                    print(f"  🧠 Memory hit! Reusing past answer (conf: {eff_conf:.3f}, thresh: {memory_threshold})")
                    
                    # Metric
                    memory_hit_total.inc()
                    from app.metrics.prometheus import memory_short_circuit_confidence_avg
                    memory_short_circuit_confidence_avg.observe(eff_conf)
                    
                    # Cache this memory hit in Redis for next time
                    if hasattr(collector, 'cache_response'):
                         await collector.cache_response(query, match["response"])

                    response_data.update({
                        "mode": "memory",
                        "response": match["response"],
                        "similarity": match["similarity"],
                        "confidence": eff_conf,
                        "memory_short_circuit": True,
                        "original_query": match["query"],
                        "source": "memory",
                        "intent": predicted_intent,
                        "refused": False
                    })
                    # Guardrail: Reusing past high-confidence answers to bypass expensive RAG or Model inference.
                    return response_data
                else:
                    print(f"  🧠 Memory match found but confidence too low ({eff_conf:.3f} < {memory_threshold})")
                    memory_miss_total.inc()
            else:
                memory_miss_total.inc()
        except Exception as e:
            # Non-blocking failure
            print(f"Memory lookup failed: {e}")
            
        # 3. Routing & Execution
        
        # Path A: External Search (RAG)
        if predicted_intent == "external_search":
            # 1. Check Memory (Feedback Retriever) for past high-confidence answer
            # Use sync search since we are likely missing hot cache
            matches = memory.search_similar(query, top_k=1, min_similarity=0.85)
            
            if matches:
                match = matches[0]
                print(f"  🧠 Memory hit! Reusing past answer (similarity: {match['similarity']})")
                await collector.cache_response(query, match["response"])
                response_data.update({
                    "mode": "rag-memory",
                    "response": match["response"],
                    "memory_used": True,
                    "similarity": match["similarity"],
                    "confidence": match.get("confidence", 0.9), # Fallback high conf for memory
                    "source": "memory",
                    "intent": "external_search",
                    "refused": False
                })
                return response_data

            # 2. Continue with external RAG if no memory hit
            # Security: RAG Capability Guard
            import os
            if not os.getenv("TAVILY_API_KEY"):
                print("  ❌ RAG unavailable: TAVILY_API_KEY not configured")
                response_data.update({
                    "mode": "refused",
                    "response": "I cannot search for external information at this time. This capability is currently unavailable.",
                    "confidence": 0.0,
                    "source": "refused",
                    "intent": "external_search",
                    "refused": True
                })
                return response_data
            
            try:
                # Security: Hard Limit on RAG Retrieval Time
                context, citations = await asyncio.wait_for(run_sync(search_web_context, query), timeout=8.0)
            except asyncio.TimeoutError:
                print(f"  ❌ RAG Timeout (8s limit exceeded)")
                response_data.update({
                    "mode": "refused",
                    "response": "External search timed out. The service is currently experiencing high latency.",
                    "confidence": 0.0,
                    "source": "refused",
                    "intent": "external_search",
                    "refused": True
                })
                return response_data

            # Synthesize with reasoner (Ollama)
            final_response = await run_sync(self.reasoner.synthesize_with_context, query, context)
            
            # Response Completeness Guard
            if self._is_incomplete(final_response):
                final_response += "\n\n(This answer may be incomplete. You can ask a follow-up.)"
            
            # Citation Integrity Guard
            final_response = self._clean_citations(final_response, citations)
            
            # Log for continuous learning (A+B=AB)
            asyncio.create_task(collector.log_interaction(
                query=query,
                context=context,
                response=final_response,
                intent="rag-external",
                source="rag"
            ))
            
            # Cache result
            await collector.cache_response(query, final_response)
            
            response_data.update({
                "mode": "rag-external",
                "response": final_response,
                "context_used": True,
                "confidence": 1.0, # RAG is considered high confidence by design intent
                "source": "rag",
                "intent": "external_search",
                "refused": False,
                "citations": citations
            })
            return response_data
            
        # Path B: Complex Reasoning (Chain of Thought)
        elif predicted_intent == "complex_reasoning":
            reasoning_result = await run_sync(self.reasoner.reason, query)
            
            response_data.update({
                "mode": "internal-reasoning",
                "response": reasoning_result["answer"],
                "reasoning_trace": reasoning_result["reasoning"],
                "reasoning_used": True,
                "confidence": 1.0,
                "source": "model",
                "intent": "complex_reasoning",
                "refused": False
            })
            return response_data
            
        # Path C: Simple Internal (Adapter/Base)
        else: # simple_internal or fallback
            # Semantic router still useful for domain selection within 'simple' intent
            from app.routing.semantic_router import get_semantic_router
            router = get_semantic_router()
            domain, conf = router.classify(query)
            
            if self.adapter_mgr.has_adapter(domain):
                resp = await run_sync(self.adapter_mgr.generate_with_adapter, domain, query)
                response_data.update({
                    "mode": "adapter",
                    "domain": domain,
                    "response": resp,
                    "adapter_used": True,
                    "confidence": conf,
                    "source": "model",
                    "intent": "simple_internal", # adapter is subtype
                    "refused": False
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
                draft_response, confidence = await run_sync(estimator.estimate_confidence, query, generate_fn)

                
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
                    
                    # Cache confident answer
                    await collector.cache_response(query, draft_response)
                    
                    # Log confident interaction
                    asyncio.create_task(collector.log_interaction(
                        query=query,
                        context="internal-knowledge",
                        response=draft_response,
                        intent="internal-confident",
                        confidence=confidence,
                        source="model"
                    ))

                    response_data.update({
                        "mode": "internal-confident",
                        "response": draft_response,
                        "confidence": confidence,
                        "source": "model",
                        "intent": "simple_internal",
                        "refused": False
                    })
                    return response_data
                
                # LOW CONFIDENCE - Check retrieval eligibility
                print(f"  ⚠️ Low confidence - checking retrieval eligibility")
                
                if get_retrieval_gate().should_retrieve(query):
                    # Novel query - trigger retrieval
                    print(f"  🔍 Novel query detected - checking memory first")
                    
                    # 1. Check Memory (Feedback Retriever)
                    matches = memory.search_similar(query, top_k=1, min_similarity=0.85)
                    
                    if matches:
                        match = matches[0]
                        print(f"  🧠 Memory hit! Reusing past answer (similarity: {match['similarity']})")
                        await collector.cache_response(query, match["response"])
                        response_data.update({
                            "mode": "rag-memory",
                            "response": match["response"],
                            "memory_used": True,
                            "similarity": match["similarity"],
                            "confidence": match.get("confidence", 0.85),
                            "source": "memory",
                            "intent": "simple_internal", # fallback from epistemic
                            "refused": False
                        })
                        return response_data

                    # 2. Trigger active retrieval
                    print(f"  🔍 Triggering external retrieval")
                    gate_decision_total.labels(type="search").inc()
                    
                    context = await run_sync(search_web_context, query)
                    
                    # Synthesize with reasoner (Ollama)
                    synthesized = await run_sync(self.reasoner.synthesize_with_context, query, context)
                    
                    inference_latency.labels(mode="search").observe(time.time() - start_time)
                    
                    # Log RAG interaction
                    asyncio.create_task(collector.log_interaction(
                        query=query,
                        context=context,
                        response=synthesized,
                        intent="rag-fallback",
                        source="rag"
                    ))
                    
                    # Cache it
                    await collector.cache_response(query, synthesized)

                    response_data.update({
                        "mode": "rag",
                        "response": synthesized,
                        "context": context,
                        "confidence": confidence,
                        "retrieved": True,
                        "source": "rag",
                        "intent": "simple_internal",
                        "refused": False
                    })
                    return response_data
                
                # LOW NOVELTY + LOW CONFIDENCE -> FORCED ABSTENTION
                # Guardrail: Epistemic gating logic enforces forced abstention for low-confidence, known-frontier queries to prevent hallucinations.
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
                    "abstained": True,
                    "source": "refused",
                    "intent": "simple_internal",
                    "refused": True
                })
                return response_data

# Global instance
_orchestrator = None

def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
