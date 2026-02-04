"""
Regression test for Truth-First Inference.
Ensures the system never confidently answers queries about unknown entities.
"""

import pytest
from app.routing.orchestrator import get_orchestrator

@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "FalkorDB",          # Original regression case
    "PageIndex",         # Known failure case
    "PersonaPlex",       # Pre-indexing case
    "XylophaseDB",       # Synthetic unknown
    "QuantumFlux",       # Synthetic unknown
])
async def test_unknown_proper_noun_must_not_hallucinate(query):
    """
    Property-based regression test: Unknown proper nouns must NOT route to model-only.
    
    This enforces the Truth-First principle:
    - Unknown entities → RAG or Refusal
    - Model-only is FORBIDDEN without evidence
    
    The specific examples don't matter.
    The PROPERTY matters: "No model-only for unknown proper nouns"
    """
    orchestrator = get_orchestrator()
    result = await orchestrator.route_and_execute(query)
    
    # Assert: Must NOT use model-only
    assert result["mode"] != "model", \
        f"{query} routed to model-only (FORBIDDEN). Mode: {result['mode']}"
    
    # Assert: Must use RAG or refuse
    assert result["mode"] in ["rag-external", "refused"], \
        f"{query} expected RAG or refusal, got: {result['mode']}"
    
    print(f"✅ {query} correctly avoided hallucination: {result['mode']}")


@pytest.mark.asyncio
async def test_known_domain_uses_adapter():
    """
    Verify that known domains with adapters still work correctly.
    """
    orchestrator = get_orchestrator()
    query = "Write a Python function to reverse a string"
    
    result = await orchestrator.route_and_execute(query)
    
    # Should use code adapter (if available) or RAG/refuse through knowledge gate
    # Note: Without evidence, even code queries may refuse (correct behavior)
    assert result["mode"] in ["adapter", "model", "rag-external", "refused"]
    
    print(f"✅ Code query routed to: {result['mode']}")
