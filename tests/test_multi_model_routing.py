"""Tests for cost/latency-aware multi-model routing behind the Reasoner factory."""

import os

os.environ["USE_DETERMINISTIC_INFERENCE"] = "true"

import pytest

from app.reasoners.base import Reasoner
from app.reasoners.deterministic import DeterministicReasoner
from app.reasoners.router import (
    DEFAULT_REGISTRY,
    ModelSpec,
    RoutingReasoner,
    estimate_complexity_tier,
    load_registry,
    select_model,
)
from app.reasoners import factory


class StubReasoner(Reasoner):
    def __init__(self, tag="stub"):
        self.tag = tag

    async def infer(self, prompt):
        return f"{self.tag}:{prompt}"

    async def reason(self, query):
        return {"answer": f"{self.tag}:{query}"}


class TestComplexityEstimation:
    def test_short_factual_is_tier1(self):
        assert estimate_complexity_tier("capital of France?") == 1

    def test_reasoning_marker_is_tier2(self):
        assert estimate_complexity_tier("why is the sky blue?") == 2

    def test_long_prompt_is_tier3(self):
        assert estimate_complexity_tier("word " * 200) == 3

    def test_double_markers_are_tier3(self):
        assert estimate_complexity_tier("explain and compare X vs Y step by step") == 3


class TestSelection:
    SPECS = [
        ModelSpec(name="tiny", tier=1, cost_per_1k_tokens=0.0, typical_latency_ms=10),
        ModelSpec(name="mid", tier=2, cost_per_1k_tokens=0.5, typical_latency_ms=500),
        ModelSpec(name="big", tier=3, cost_per_1k_tokens=2.0, typical_latency_ms=2000),
    ]

    def test_cheapest_sufficient_model_wins(self):
        assert select_model(self.SPECS, 1).name == "tiny"
        assert select_model(self.SPECS, 2).name == "mid"
        assert select_model(self.SPECS, 3).name == "big"

    def test_no_eligible_raises(self):
        with pytest.raises(ValueError):
            select_model([self.SPECS[0]], 2)

    def test_tie_broken_by_latency(self):
        specs = [
            ModelSpec(name="a", tier=1, cost_per_1k_tokens=1.0, typical_latency_ms=900),
            ModelSpec(name="b", tier=1, cost_per_1k_tokens=1.0, typical_latency_ms=200),
        ]
        assert select_model(specs, 1).name == "b"


class TestRoutingReasoner:
    def _router(self, specs=None):
        backends = {"fast": StubReasoner("fast"), "slow": StubReasoner("slow")}
        specs = specs or [
            ModelSpec(name="fast", tier=1, cost_per_1k_tokens=0.0, typical_latency_ms=50),
            ModelSpec(name="slow", tier=3, cost_per_1k_tokens=9.9, typical_latency_ms=900),
        ]
        return RoutingReasoner(backends, specs=specs)

    async def test_simple_query_routes_to_cheapest(self):
        router = self._router()
        out = await router.infer("hi?")
        assert out.startswith("fast:")
        assert router.last_selection == "fast"

    async def test_complex_query_routes_to_capable(self):
        router = self._router()
        await router.infer("explain in depth why A, then compare B step by step")
        assert router.last_selection == "slow"

    async def test_reason_dispatches(self):
        result = await self._router().reason("what is 2+2?")
        assert result["answer"].startswith("fast:")

    async def test_is_a_reasoner(self):
        assert isinstance(self._router(), Reasoner)

    async def test_unsatisfiable_tier_uses_fallback(self):
        # tier-1 backend only, but a long reasoning-heavy prompt requires tier 3
        router = self._router(specs=[ModelSpec(name="fast", tier=1, cost_per_1k_tokens=0.0, typical_latency_ms=50)])
        router.fallback = StubReasoner("fallback")
        out = await router.infer("word " * 200)
        assert out.startswith("fallback:")


class TestRegistryAndFactory:
    def test_default_registry_loads(self):
        names = [s.name for s in load_registry()]
        assert "deterministic" in names

    def test_json_registry_override(self, tmp_path):
        cfg = tmp_path / "models.json"
        cfg.write_text(
            '{"models": [{"name": "mini", "tier": 1, "cost_per_1k_tokens": 0.1, "typical_latency_ms": 300}]}'
        )
        specs = load_registry(str(cfg))
        assert specs[0].name == "mini"

    async def test_factory_opt_in_returns_router(self, monkeypatch):
        # deterministic mode short-circuits the factory before routing; clear it
        monkeypatch.delenv("USE_DETERMINISTIC_INFERENCE", raising=False)
        monkeypatch.delenv("USE_MOCKED_MODELS", raising=False)
        monkeypatch.setenv("MODEL_ROUTING_ENABLED", "true")
        monkeypatch.delenv("MODEL_ROUTING_CONFIG", raising=False)
        factory.reset_reasoner()
        reasoner = factory.get_reasoner()
        assert isinstance(reasoner, RoutingReasoner)
        # simple prompt should hit the zero-cost deterministic backend
        answer = await reasoner.infer("hello there?")
        assert answer  # deterministic backend produced output
        factory.reset_reasoner()
        monkeypatch.delenv("MODEL_ROUTING_ENABLED")

    async def test_factory_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("MODEL_ROUTING_ENABLED", raising=False)
        monkeypatch.setenv("USE_DETERMINISTIC_INFERENCE", "true")
        factory.reset_reasoner()
        assert isinstance(factory.get_reasoner(), DeterministicReasoner)
        factory.reset_reasoner()
