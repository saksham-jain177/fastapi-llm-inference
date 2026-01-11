"""
Runtime Factory for Reasoner instantiation.
Single responsibility: decide which backend to use at runtime.

No inference logic. Vendor imports only when that branch executes.
"""

import os
from typing import TYPE_CHECKING

from app.reasoners.base import Reasoner
from app.reasoners.deterministic import DeterministicReasoner

if TYPE_CHECKING:
    from app.reasoners.ollama_backend import OllamaReasoner


# Singleton instance
_reasoner_instance: Reasoner = None


def get_reasoner() -> Reasoner:
    """
    Get the appropriate Reasoner instance based on environment configuration.

    Selection logic:
    1. If USE_DETERMINISTIC_INFERENCE=true -> DeterministicReasoner
    2. Else if provider == ollama -> OllamaReasoner
    3. Else -> DeterministicReasoner (fallback)

    Returns:
        Concrete Reasoner instance
    """
    global _reasoner_instance

    if _reasoner_instance is not None:
        return _reasoner_instance

    # Check environment for deterministic mode
    use_deterministic = os.getenv("USE_DETERMINISTIC_INFERENCE", "false").lower() == "true"
    
    # Legacy support: also check old variable name
    if os.getenv("USE_MOCKED_MODELS", "false").lower() == "true":
        use_deterministic = True

    if use_deterministic:
        print("🔧 Using DeterministicReasoner (hardware-independent mode)")
        _reasoner_instance = DeterministicReasoner()
        return _reasoner_instance

    # Determine provider
    provider = os.getenv("INFERENCE_PROVIDER", "ollama").lower()

    if provider == "ollama":
        try:
            # Import only when this branch executes
            from app.reasoners.ollama_backend import OllamaReasoner
            print("🦙 Using OllamaReasoner")
            _reasoner_instance = OllamaReasoner()
            return _reasoner_instance
        except ImportError as e:
            print(f"⚠️ Ollama backend unavailable ({e}), falling back to DeterministicReasoner")
            _reasoner_instance = DeterministicReasoner()
            return _reasoner_instance

    # Default fallback
    print("🔧 Using DeterministicReasoner (default fallback)")
    _reasoner_instance = DeterministicReasoner()
    return _reasoner_instance


def reset_reasoner():
    """Reset the singleton instance. Used for testing."""
    global _reasoner_instance
    _reasoner_instance = None
