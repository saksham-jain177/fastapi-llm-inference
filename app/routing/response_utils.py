import re

def is_incomplete(text: str) -> bool:
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

def clean_citations(text: str, citations: list) -> str:
    """
    Enforce citation integrity:
    1. If no citations exist, remove [Source X] markers (hallucinations).
    2. If citations exist but text has no markers, that's fine (frontend handles it).
    """
    # Case 1: No valid citations -> Strip all markers
    if not citations:
        # Matches [Source 1], [Source 10], etc.
        return re.sub(r'\[Source \d+\]', '', text).strip()
        
    return text
