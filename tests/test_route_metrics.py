"""Tests for per-route latency histograms on the Orchestrator."""

import os

os.environ["USE_DETERMINISTIC_INFERENCE"] = "true"

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.metrics.route_metrics import route_total, _normalize_route
from app.routing.orchestrator import Orchestrator


def _get_route_counts():
    return {
        sample.labels["route"]: sample.value
        for sample in route_total.collect()[-1].samples
        if sample.name == "orchestrator_route_total"
    }


def _mock_pipeline(router_classify=("general", 0.9), reasoner_answer="A complete answer."):
    """Return a context manager patching the orchestrator's collaborators."""
    return (
        patch("app.routing.orchestrator.search_web_context", return_value=("Context", [])),
        patch(
            "app.routing.orchestrator.get_semantic_router",
            return_value=MagicMock(classify=MagicMock(return_value=router_classify)),
        ),
        patch("app.routing.orchestrator.get_reasoner"),
        patch("app.routing.orchestrator.get_data_collector"),
    )


class TestRouteNormalization:
    def test_mode_key(self):
        assert _normalize_route({"mode": "rag-external"}) == "rag-external"

    def test_missing_mode_is_unknown(self):
        assert _normalize_route({}) == "unknown"


class TestInstrumentation:
    async def test_refusal_path_recorded(self):
        before = _get_route_counts().get("refused", 0)
        orch = Orchestrator()
        result = await orch.route_and_execute("")
        assert result["refused"] is True
        after = _get_route_counts().get("refused", 0)
        assert after == before + 1

    async def test_rag_path_recorded(self):
        search_patch, router_patch, reasoner_patch, collector_patch = _mock_pipeline()
        with search_patch as mock_search, router_patch, \
             reasoner_patch as mock_get_reasoner, collector_patch as mock_get_collector, \
             patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            mock_reasoner = AsyncMock()
            mock_reasoner.synthesize_with_context.return_value = "A complete answer."
            mock_get_reasoner.return_value = mock_reasoner
            mock_collector = AsyncMock()
            mock_collector.get_cached_response.return_value = None
            mock_get_collector.return_value = mock_collector

            before = _get_route_counts().get("rag-external", 0)
            orch = Orchestrator()
            result = await orch.route_and_execute("What is the capital of France?")
            assert result["mode"] == "rag-external"
            assert _get_route_counts().get("rag-external", 0) == before + 1

    async def test_search_failure_degrades_to_refused_route(self):
        """RAG failures are caught inside _execute_external_rag → 'refused' route."""
        search_patch, router_patch, _, collector_patch = _mock_pipeline()
        with search_patch as mock_search, router_patch, \
             patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            mock_search.side_effect = RuntimeError("search down")
            with collector_patch as mock_get_collector:
                mock_collector = AsyncMock()
                mock_collector.get_cached_response.return_value = None
                mock_get_collector.return_value = mock_collector

                orch = Orchestrator()
                result = await orch.route_and_execute("some query")
                assert result["mode"] == "refused"
                assert "refused" in _get_route_counts()
