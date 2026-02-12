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
    Verify that known domains with adapters explicitly use the adapter path.
    """
    from unittest.mock import patch, MagicMock
    
    # Force the adapter manager to say "YES" for this domain
    with patch("app.models.adapter_manager.get_adapter_manager") as mock_get_mgr:
        mock_mgr = MagicMock()
        mock_mgr.has_adapter.return_value = True
        mock_mgr.generate_with_adapter.return_value = "Adapter Output"
        mock_get_mgr.return_value = mock_mgr
        
        # Patch router to return a specific domain
        with patch("app.routing.orchestrator.get_semantic_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.classify.return_value = ("coding", 0.95)
            mock_get_router.return_value = mock_router

            # Execute
            orchestrator = get_orchestrator()
            # We must recreate orchestrator or patch the one used, but get_orchestrator is cached
            # Ideally we patch where it is used. Orchestrator uses get_adapter_manager at init.
            # So we need to patch the Orchestrator's instance of adapter_mgr if it's already created.
            
            # Better approach: Instantiate a fresh orchestrator with mocked dependencies
            # But get_orchestrator is a singleton.
            # Let's adjust attributes on the singleton for the test.
            
            orchestrator.adapter_mgr = mock_mgr
            # orchestrator.reasoner is already mockable/swappable
            
            query = "Write a Python function"
            result = await orchestrator.route_and_execute(query)
            
            assert result["mode"] == "adapter"
            assert result["response"] == "Adapter Output"
            assert result["confidence"] == 0.95
            
    print(f"✅ Adapter path explicitly verified.")
