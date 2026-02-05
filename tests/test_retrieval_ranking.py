import pytest
from app.rag.retrieval import compute_lexical_relevance
from app.rag.context_manager import ContextPacker

def test_lexical_relevance_overlap():
    query = "Capital of France"
    high_relevance = "The capital of France is Paris."
    low_relevance = "France is a country in Europe."
    no_relevance = "Italy has a nice climate."
    
    score_high = compute_lexical_relevance(query, high_relevance)
    score_low = compute_lexical_relevance(query, low_relevance)
    score_none = compute_lexical_relevance(query, no_relevance)
    
    assert score_high > score_low
    assert score_low > score_none
    assert score_none == 0.0

def test_lexical_relevance_boilerplate_penalty():
    query = "data privacy"
    # Content with high diversity
    good_content = "This document discusses data privacy and security in modern cloud systems."
    # Content with low diversity (repeating 'the' and 'and') - boilerplate
    bad_content = "the " * 50 + "data privacy " + "and " * 50
    
    score_good = compute_lexical_relevance(query, good_content)
    score_bad = compute_lexical_relevance(query, bad_content)
    
    # Bad content should be penalized despite having the keywords
    assert score_bad < score_good * 0.6

def test_context_packer_ranking():
    packer = ContextPacker()
    query = "FalkorDB"
    results = [
        {"title": "Low", "content": "Some generic database stuff", "url": "url1"},
        {"title": "High", "content": "FalkorDB is a graph database", "url": "url2"},
        {"title": "Medium", "content": "Graph databases like FalkorDB and others", "url": "url3"},
    ]
    
    packed_context, used_citations = packer.pack(query, results)
    
    # "High" should be first in the context
    assert "Source 1: High" in packed_context
    assert "FalkorDB is a graph database" in packed_context
    assert packed_context.index("High") < packed_context.index("Medium")
    assert packed_context.index("Medium") < packed_context.index("Low")
    
    # Citations should match the order in context
    assert used_citations[0]["title"] == "High"
    assert used_citations[1]["title"] == "Medium"
    assert used_citations[2]["title"] == "Low"
