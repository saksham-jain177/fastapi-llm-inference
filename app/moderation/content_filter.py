"""
Content moderation for filtering harmful, offensive, or inappropriate queries.
"""

from better_profanity import profanity
from typing import Tuple
import re


class ContentModerator:
    """
    Multi-layer content moderation system:
    - Profanity detection
    - Harmful content patterns
    - Prompt injection attempts
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
        """Initialize profanity filter."""
        profanity.load_censor_words()
    
    def check_profanity(self, text: str) -> bool:
        """Check if text contains profanity."""
        return profanity.contains_profanity(text)
    
    def check_harmful_patterns(self, text: str) -> bool:
        """Check if text matches harmful content patterns."""
        text_lower = text.lower()
        for pattern in self.HARMFUL_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def check_prompt_injection(self, text: str) -> bool:
        """Check if text contains prompt injection attempts."""
        text_lower = text.lower()
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def moderate(self, text: str) -> Tuple[bool, str]:
        """
        Moderate text and return (is_safe, reason).
        
        Args:
            text: Text to moderate
            
        Returns:
            Tuple of (is_safe: bool, reason: str)
        """
        # Check profanity
        if self.check_profanity(text):
            return False, "Content contains inappropriate language"
        
        # Check harmful patterns
        if self.check_harmful_patterns(text):
            return False, "Content appears to request harmful or illegal information"
        
        # Check prompt injection
        if self.check_prompt_injection(text):
            return False, "Prompt injection attempt detected"
        
        return True, "Content passed moderation"


# Global instance
_moderator = None


def get_moderator() -> ContentModerator:
    """Get or create content moderator instance."""
    global _moderator
    if _moderator is None:
        _moderator = ContentModerator()
    return _moderator
