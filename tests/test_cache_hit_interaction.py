"""
Regression tests for the semantic cache × streaming interaction.

The Orchestrator's Redis cache-hit path previously returned early WITHOUT
calling collector.log_interaction, so every served-from-cache answer was
silently dropped from the RLHF/calibration dataset. These tests pin the fixed,
symmetric behavior.
"""

import pytest

from app.routing.orchestrator import Orchestrator


class FakeCollector:
    """Minimal DataCollector stand-in: fake Redis cache, recorded logs."""

    def __init__(self, cached_response=None):
        self.redis_client = None
        self.mongo_collection = None
        self.cached_response = cached_response
        self.logged = []
        self.cached = []

    async def get_cached_response(self, query):
        return self.cached_response

    async def cache_response(self, query, response, ttl=300):
        self.cached.append((query, response))

    async def log_interaction(self, **kwargs):
        self.logged.append(kwargs)


@pytest.fixture()
def orchestrator(monkeypatch):
    monkeypatch.setenv("USE_DETERMINISTIC_INFERENCE", "true")
    import app.routing.orchestrator as orch_mod
    collector = FakeCollector(cached_response="Cached answer about redis.")
    monkeypatch.setattr(orch_mod, "get_data_collector", lambda: collector)
    orch_mod._orchestrator = None
    yield orch_mod.get_orchestrator(), collector
    orch_mod._orchestrator = None


class TestCacheHitLogging:
    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_response(self, orchestrator):
        orch, collector = orchestrator
        result = await orch.route_and_execute("explain redis caching strategies")
        assert result["mode"] == "redis_cache"
        assert result["cache_hit"] is True
        assert result["response"] == "Cached answer about redis."
        assert result["source"] == "redis"

    @pytest.mark.asyncio
    async def test_cache_hit_is_still_logged(self, orchestrator):
        """Regression: the early return must not skip interaction logging."""
        orch, collector = orchestrator
        await orch.route_and_execute("explain redis caching strategies")
        assert len(collector.logged) == 1
        entry = collector.logged[0]
        assert entry["query"] == "explain redis caching strategies"
        assert entry["response"] == "Cached answer about redis."
        assert entry["source"] == "redis_cache"

    @pytest.mark.asyncio
    async def test_cache_hit_logged_once_not_doubled(self, orchestrator):
        """The hit path must log exactly once (no duplicate via fallthrough)."""
        orch, collector = orchestrator
        await orch.route_and_execute("explain redis caching strategies")
        assert len(collector.logged) == 1

    @pytest.mark.asyncio
    async def test_cache_miss_falls_through_to_pipeline(self, orchestrator):
        """Miss keeps existing pipeline behavior (KB/gate path runs)."""
        import app.routing.orchestrator as orch_mod
        collector = FakeCollector(cached_response=None)
        orch_mod.get_data_collector = lambda: collector
        try:
            result = await orch_mod.get_orchestrator().route_and_execute(
                "explain redis caching strategies"
            )
            # No cache fields on a miss; pipeline continued past the lookup.
            assert "cache_hit" not in result or result.get("cache_hit") is not True
        finally:
            del orch_mod.get_data_collector
