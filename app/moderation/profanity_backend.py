import re
from typing import Tuple
from app.moderation.base import Moderator

class ProfanityModerator(Moderator):
    """
    Production-grade moderation backend using better-profanity.
    Implements lazy-loading of the dependency to avoid import-time coupling.
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
    
    def __init__(self):
        # Lazy import of better_profanity
        try:
            from better_profanity import profanity
            self._profanity = profanity
            self._profanity.load_censor_words()
        except ImportError:
            # This should ideally not happen if factory selection is correct,
            # but we handle it gracefully here if it does.
            self._profanity = None
    
    def moderate(self, text: str) -> Tuple[bool, str]:
        if self._profanity and self._profanity.contains_profanity(text):
            return False, "Content contains inappropriate language"
            
        text_lower = text.lower()
        
        # Harmful patterns
        for pattern in self.HARMFUL_PATTERNS:
            if re.search(pattern, text_lower):
                return False, "Content appears to request harmful or illegal information"
        
        # Prompt injection
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                return False, "Prompt injection attempt detected"
        
        return True, "Content passed moderation"
