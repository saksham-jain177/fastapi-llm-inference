"""
Cost/latency-aware multi-model routing.

Sits BEHIND the existing Reasoner factory interface: `get_reasoner()` may now
return a `RoutingReasoner`, which is itself a Reasoner. Callers (Orchestrator,
endpoints) are unchanged — they still see `.infer() / .reason() /
.synthesize_with_context()`.

Selection policy (deliberately simple, no learned scorer):
- Each candidate has a static spec: relative cost per 1k tokens, typical
  latency ms, and a capability tier.
- The prompt's complexity is estimated from cheap signals (length, reasoning
  markers like "why"/"explain"/"step", multi-part questions).
- Pick the CHEAPEST candidate whose tier satisfies the complexity requirement;
  break ties by latency. This keeps cost minimal without over-engineering.

Env config:
- MODEL_ROUTING_ENABLED=true   → activate routing in the factory
- MODEL_ROUTING_CONFIG         → optional path to a JSON registry overriding
  the built-in default registry.

The default registry contains the two real backends this project ships
(deterministic = zero cost, ollama = local/free but slower). Adding cloud
models later is a registry change, not an interface change.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

from app.reasoners.base import Reasoner


@dataclass(frozen=True)
class ModelSpec:
    """Static cost/latency/capability profile of one candidate model."""
    name: str
    tier: int                      # capability tier; higher = handles harder queries
    cost_per_1k_tokens: float      # 0.0 for local/free backends
    typical_latency_ms: int        # rough single-request latency


# Built-in registry: ordered cheapest-first for readability.
DEFAULT_REGISTRY: List[ModelSpec] = [
    ModelSpec(name="deterministic", tier=1, cost_per_1k_tokens=0.0, typical_latency_ms=5),
    ModelSpec(name="ollama-small", tier=2, cost_per_1k_tokens=0.0, typical_latency_ms=800),
    ModelSpec(name="ollama-large", tier=3, cost_per_1k_tokens=0.0, typical_latency_ms=2500),
]

# Complexity tiers required by prompt classes
SIMPLE_MAX_CHARS = 120
REASONING_MARKERS = ("why", "explain", "step", "compare", "analyze", "evaluate", "design")


def estimate_complexity_tier(prompt: str) -> int:
    """
    Map a prompt to the minimum capability tier needed (pure function).

    1 = short factual lookup
    2 = normal query / moderate length
    3 = reasoning-heavy or long multi-part prompt
    """
    text = prompt.lower()
    if len(text) <= SIMPLE_MAX_CHARS and not any(m in text for m in REASONING_MARKERS):
        return 1
    marker_hits = sum(1 for m in REASONING_MARKERS if m in text)
    if len(text) > 600 or marker_hits >= 2:
        return 3
    return 2


def select_model(specs: List[ModelSpec], required_tier: int) -> ModelSpec:
    """
    Choose the cheapest spec meeting the tier; ties broken by latency.
    Raises ValueError when no candidate can satisfy the tier.
    """
    eligible = [s for s in specs if s.tier >= required_tier]
    if not eligible:
        raise ValueError(
            f"No model in registry meets required tier {required_tier} "
            f"(available tiers: {sorted({s.tier for s in specs})})"
        )
    return min(eligible, key=lambda s: (s.cost_per_1k_tokens, s.typical_latency_ms))


class RoutingReasoner(Reasoner):
    """
    Reasoner that dispatches each call to the best candidate backend.

    Args:
        backends: mapping of model name -> instantiated Reasoner.
        specs: registry entries parallel to `backends`.
        fallback: backend used if selection or dispatch fails unexpectedly.
    """

    def __init__(
        self,
        backends: Dict[str, Reasoner],
        specs: List[ModelSpec] = None,
        fallback: Reasoner = None,
    ):
        self.backends = backends
        self.specs = list(specs) if specs is not None else [
            s for s in DEFAULT_REGISTRY if s.name in backends
        ]
        self.fallback = fallback
        self.last_selection: str = None  # observability hook for tests/metrics

    def _select(self, prompt: str) -> Reasoner:
        tier = estimate_complexity_tier(prompt)
        chosen = select_model(self.specs, tier)
        self.last_selection = chosen.name
        return self.backends[chosen.name]

    async def infer(self, prompt: str) -> str:
        try:
            return await self._select(prompt).infer(prompt)
        except ValueError:
            if self.fallback is not None:
                return await self.fallback.infer(prompt)
            raise

    async def reason(self, query: str):
        try:
            return await self._select(query).reason(query)
        except ValueError:
            if self.fallback is not None:
                return await self.fallback.reason(query)
            raise


def load_registry(config_path: str = None) -> List[ModelSpec]:
    """Load the model registry from JSON, or the built-in default."""
    path = config_path or os.getenv("MODEL_ROUTING_CONFIG")
    if not path:
        return list(DEFAULT_REGISTRY)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [ModelSpec(**entry) for entry in raw["models"]]
