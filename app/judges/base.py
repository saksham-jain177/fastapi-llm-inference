"""
Abstract Judge interface.
No hardware dependencies at import time.
"""

from abc import ABC, abstractmethod
from typing import Tuple


class Judge(ABC):
    """
    Abstract interface for LLM-based classifiers (judges).
    """

    @abstractmethod
    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify a query into a domain and provide a confidence score.
        
        Args:
            query: User's input prompt
            
        Returns:
            Tuple of (domain, confidence)
        """
        pass
