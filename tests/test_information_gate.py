import pytest
from app.routing.information_gate import is_informative

def test_refused_queries():
    refused = [
        "hi",
        "hello",
        "???",
        "!!!",
        "hi there",
        "a",
        "123",
        "    ",
        "",
        "@@@",
        "abc" # length 3, 1 token, no structure, density 1.0 but length < 5
    ]
    for q in refused:
        assert is_informative(q) is False, f"Should have refused: '{q}'"

def test_allowed_technical_terms():
    allowed = [
        "PageIndex",     # MixedCase
        "fastAPI",       # internal caps
        "Redis",         # Length 5, Density 1.0 (Unique 5 / Content 5)
        "MongoDB",       # Structural (Upper internal)
        "FalkorDB",      # Structural
        "v2",            # Alphanumeric
        "GPT-4",         # Structural + Acronym
        "AI",            # Acronym
        "SQL",           # Acronym
        "RAG",           # Acronym
        "redis",         # Length 5, Density 1.0
        "mongodb",       # Length 7, Density 0.85 (Unique 6 / Content 7)
    ]
    for q in allowed:
        assert is_informative(q) is True, f"Should have allowed: '{q}'"

def test_allowed_phrases():
    allowed = [
        "What is CockroachDB",
        "How to use RAG",
        "Explain Redis persistence",
        "Compare SQL vs NoSQL"
    ]
    for q in allowed:
        assert is_informative(q) is True, f"Should have allowed: '{q}'"

def test_low_alphabetic_ratio():
    # Mostly symbols/digits without structure
    assert is_informative("12345") is False
    assert is_informative("!!! ???") is False
    # But alphanumeric structure allows it
    assert is_informative("v2") is True
    assert is_informative("Python3") is True
