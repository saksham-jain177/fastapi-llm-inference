import re

def is_informative(query: str) -> bool:
    """
    Deterministic signal density gate for Truth-First routing.
    Rejects low-information queries (greetings, noise) without hardcoding words.
    """
    if not query:
        return False
        
    query = query.strip()
    tokens = query.split()
    token_count = len(tokens)
    length = len(query)
    
    # 1. Phrases (Direct intent)
    if token_count >= 3:
        return True
        
    # 2. Structural Signal Detection
    # Internal capitalization (PascalCase/camelCase) or Alphanumeric mix
    def has_structural_signal(s):
        # Mixed case: upper character followed by something or preceded by something
        # Specifically looking for internal caps: "PageIndex", "fastAPI"
        has_internal_caps = bool(re.search(r'[a-z][A-Z]|[A-Z][a-z][A-Z]', s))
        # Alphanumeric: "v2", "GPT-4"
        has_alphanumeric = any(c.isdigit() for c in s) and any(c.isalpha() for c in s)
        return has_internal_caps or has_alphanumeric

    if any(has_structural_signal(t) for t in tokens):
        return True
        
    # 3. Acronym Detection (All caps, length >= 2)
    # Allows "AI", "SQL", "RAG"
    def is_acronym(s):
        return len(s) >= 2 and s.isupper() and s.isalpha()
        
    if any(is_acronym(t) for t in tokens):
        return True
        
    # Refusal Signals

    # Noise check: mostly non-alphabetic AND no structural signal
    alphabetic_chars = sum(1 for c in query if c.isalpha())
    alphabetic_ratio = alphabetic_chars / length if length > 0 else 0
    
    # 4. Density / Variety (allows "Redis", "redis")
    # Low entropy/repetition check: unique chars vs length
    is_dense = False
    if length >= 5:
        content = query.replace(" ", "")
        content_len = len(content)
        if content_len > 0:
            unique_chars = len(set(content.lower()))
            # Must have high variety AND at least some alphabetic content to be 'dense'
            if (unique_chars / content_len) >= 0.85 and any(c.isalpha() for c in content):
                is_dense = True

    # If it's a structural signal, it's informative even if low alpha (e.g. "v2")
    has_any_signal = any(has_structural_signal(t) for t in tokens) or any(is_acronym(t) for t in tokens) or is_dense
    
    if alphabetic_ratio < 0.5 and not has_any_signal:
        return False
        
    # Final Informative Decision
    return has_any_signal
