"""
Runtime Factory for Reasoner instantiation.
Single responsibility: decide which backend to use at runtime.

No inference logic. Vendor imports only when that branch executes.
"""

import os
from typing import TYPE_CHECKING

from app.observability.logging_setup import get_logger
from app.reasoners.base import Reasoner
from app.reasoners.deterministic import DeterministicReasoner

logger = get_logger("reasoners.factory")

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
        logger.info("Using DeterministicReasoner (hardware-independent mode)")
        _reasoner_instance = DeterministicReasoner()
        return _reasoner_instance

    # Determine provider
    provider = os.getenv("INFERENCE_PROVIDER", "ollama").lower()

    # Multi-model routing: cost/latency-aware selector behind the same
    # Reasoner interface. Opt-in via MODEL_ROUTING_ENABLED=true.
    if os.getenv("MODEL_ROUTING_ENABLED", "false").lower() == "true":
        try:
            from app.reasoners.router import RoutingReasoner, load_registry
            from app.reasoners.deterministic import DeterministicReasoner as _Det

            specs = load_registry()
            backends = {}
            for spec in specs:
                if spec.name == "deterministic":
                    backends[spec.name] = _Det()
                elif spec.name.startswith("ollama"):
                    try:
                        from app.reasoners.ollama_backend import OllamaReasoner
                        backends[spec.name] = OllamaReasoner()
                    except ImportError:
                        logger.warning("Routing: ollama backend unavailable for %r, skipping", spec.name)
            if backends:
                fallback = backends.get("deterministic", next(iter(backends.values())))
                logger.info("Using RoutingReasoner (cost/latency-aware multi-model routing)")
                _reasoner_instance = RoutingReasoner(backends, specs=specs, fallback=fallback)
                return _reasoner_instance
            logger.warning("MODEL_ROUTING_ENABLED but no backends available; falling back")
        except Exception as e:
            logger.warning("RoutingReasoner unavailable (%s); falling back", e)

    if provider == "ollama":
        try:
            # Import only when this branch executes
            from app.reasoners.ollama_backend import OllamaReasoner
            logger.info("Using OllamaReasoner")
            _reasoner_instance = OllamaReasoner()
            return _reasoner_instance
        except ImportError as e:
            logger.warning("Ollama backend unavailable (%s), falling back to DeterministicReasoner", e)
            _reasoner_instance = DeterministicReasoner()
            return _reasoner_instance

    # Default fallback
    logger.info("Using DeterministicReasoner (default fallback)")
    _reasoner_instance = DeterministicReasoner()
    return _reasoner_instance


def reset_reasoner():
    """Reset the singleton instance. Used for testing."""
    global _reasoner_instance
    _reasoner_instance = None
