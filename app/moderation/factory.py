import os
from app.moderation.base import Moderator

import threading
_moderator_lock = threading.Lock()
_moderator = None

def get_moderator() -> Moderator:
    """
    Runtime factory for selecting the moderation backend.
    Prioritizes DeterministicModerator if USE_DETERMINISTIC_INFERENCE is set.
    """
    global _moderator
    if _moderator is not None:
        return _moderator
        
    with _moderator_lock:
        if _moderator is not None:
            return _moderator
            
        use_deterministic = os.getenv("USE_DETERMINISTIC_INFERENCE", "false").lower() == "true"
        
        if use_deterministic:
            from app.moderation.deterministic import DeterministicModerator
            _moderator = DeterministicModerator()
        else:
            try:
                from app.moderation.profanity_backend import ProfanityModerator
                _moderator = ProfanityModerator()
            except Exception:
                # Fallback to deterministic if backend loading fails
                from app.moderation.deterministic import DeterministicModerator
                _moderator = DeterministicModerator()
                
        return _moderator

def reset_moderator():
    """Reset the global moderator instance (primarily for testing)."""
    global _moderator
    with _moderator_lock:
        _moderator = None
