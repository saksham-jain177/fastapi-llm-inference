import re
from typing import Tuple
from app.moderation.base import Moderator

class DeterministicModerator(Moderator):
    """
    Hardware-independent moderation backend.
    Uses regex and simple keyword matching without third-party dependencies.
    Safe for CI and staging environments.
    """
    
    # Harmful patterns (regex-based)
    HARMFUL_PATTERNS = [
        r'\b(how to (hack|exploit|bypass|crack))\b',
        r'\b(illegal|unlawful) (activity|content)\b',
        r'\b(create|make|build) (virus|malware|exploit)\b',
        r'\b(bypass|circumvent) (security|authentication)\b',
    ]
    
    # Prompt injection patterns
    INJECTION_PATTERNS = [
        r'ignore (previous|all) instructions?',
        r'disregard (the )?(system|above) prompt',
        r'you are now',
        r'from now on',
        r'pretend (to be|you are)',
    ]
    
    # Simple deterministic profanity keywords (placeholder list)
    PROFANE_KEYWORDS = {"profane_token_1", "profane_token_2", "harmful_content_trigger"}
    
    def moderate(self, text: str) -> Tuple[bool, str]:
        text_lower = text.lower()
        
        # 1. Simple Keyword Profanity (Placeholder)
        for word in self.PROFANE_KEYWORDS:
            if word in text_lower:
                return False, "Content contains inappropriate language (Deterministic)"
        
        # 2. Harmful patterns
        for pattern in self.HARMFUL_PATTERNS:
            if re.search(pattern, text_lower):
                return False, "Content appears to request harmful or illegal information"
        
        # 3. Prompt injection
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                return False, "Prompt injection attempt detected"
        
        return True, "Content passed deterministic moderation"
