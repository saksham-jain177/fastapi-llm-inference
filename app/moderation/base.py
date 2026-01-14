from abc import ABC, abstractmethod
from typing import Tuple

class Moderator(ABC):
    """
    Abstract interface for content moderation.
    Ensures consistent behavior across different moderation backends.
    """
    
    @abstractmethod
    def moderate(self, text: str) -> Tuple[bool, str]:
        """
        Moderate text and return (is_safe, reason).
        
        Args:
            text: Text to moderate
            
        Returns:
            Tuple of (is_safe: bool, reason: str)
        """
        pass
