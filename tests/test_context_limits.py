import pytest
from app.rag.context_manager import ContextPacker

def test_context_packer_budget_truncation():
    # Small budget to force truncation
    packer = ContextPacker(char_budget=100)
    query = "test"
    results = [
        {"title": "Long result", "content": "This is a very long result that will definitely exceed the budget of 100 characters by a significant margin.", "url": "url1"}
    ]
    
    packed_context, used_citations = packer.pack(query, results)
    
    # Must be within budget (with small margin for formatting)
    assert len(packed_context) <= 125 # Adjusted margin
    assert "..." in packed_context
    assert len(used_citations) == 1

def test_context_packer_multiple_results_budget():
    # Budget that allows only one result
    packer = ContextPacker(char_budget=150)
    query = "test"
    results = [
        {"title": "Result 1", "content": "This content is approximately 50 chars.", "url": "url1"},
        {"title": "Result 2", "content": "This content is also approximately 50 chars.", "url": "url2"},
        {"title": "Result 3", "content": "This one will push it over the limit.", "url": "url3"},
    ]
    
    packed_context, used_citations = packer.pack(query, results)
    
    # Check that Result 3 is omitted if 1 and 2 already consumed budget
    # Calculations: 
    # S1: "Source 1: Result 1\nThis content is approximately 50 chars.\nURL: url1" -> ~80 chars
    # S2: "Source 2: Result 2\nThis content is also approximately 50 chars.\nURL: url2" -> ~85 chars
    # Total: ~165 -> Result 2 might also be cut if budget is 150.
    
    assert len(used_citations) < 3
    assert "Result 1" in packed_context
    if "Result 2" in packed_context:
        assert "Result 3" not in packed_context

def test_context_packer_empty_results():
    packer = ContextPacker()
    packed_context, used_citations = packer.pack("query", [])
    assert "No relevant context" in packed_context
    assert used_citations == []
