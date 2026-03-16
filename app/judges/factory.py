"""
Runtime Factory for Judge instantiation.
"""

import os
from app.judges.base import Judge
from app.judges.deterministic import DeterministicJudge

import threading
_judge_lock = threading.Lock()
_judge_instance: Judge = None

def get_judge() -> Judge:
    """
    Get the appropriate Judge instance based on environment.
    """
    global _judge_instance
    if _judge_instance is not None:
        return _judge_instance

    with _judge_lock:
        if _judge_instance is not None:
            return _judge_instance

        use_deterministic = os.getenv("USE_DETERMINISTIC_INFERENCE", "false").lower() == "true"
        
        if use_deterministic:
            _judge_instance = DeterministicJudge()
            return _judge_instance

        # Default to Ollama if hardware independent mode is not requested
        try:
            from app.judges.ollama_backend import OllamaJudge
            _judge_instance = OllamaJudge()
            return _judge_instance
        except (ImportError, Exception):
            # Fallback to deterministic if Ollama backend cannot be loaded
            _judge_instance = DeterministicJudge()
            return _judge_instance

def reset_judge():
    """Reset the singleton instance (primarily for tests)."""
    global _judge_instance
    with _judge_lock:
        _judge_instance = None
