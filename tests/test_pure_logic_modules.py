"""
Targeted unit tests for previously-uncovered pure-logic modules.

Covers:
- app.judges (base contract, deterministic classify, factory incl. fallback)
- app.models.calibration (threshold computation, bootstrap, persistence)
- app.rag.search_tool (strict_search found/not-found paths)
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.judges.base import Judge
from app.judges.deterministic import DeterministicJudge
from app.judges import factory as judge_factory
from app.models import calibration
from app.rag.search_tool import SearchTool, get_search_tool


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------

class TestJudgeBase:
    def test_deterministic_judge_is_judge(self):
        assert isinstance(DeterministicJudge(), Judge)

    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Judge()


class TestDeterministicJudge:
    def setup_method(self):
        self.judge = DeterministicJudge()

    def test_code_domain(self):
        domain, conf = self.judge.classify("def foo(): pass")
        assert domain == "code"
        assert conf == 0.95

    def test_medical_domain(self):
        domain, conf = self.judge.classify("patient diagnosis guidelines")
        assert domain == "medical"
        assert conf == 0.95

    def test_legal_domain(self):
        domain, conf = self.judge.classify("contract law question")
        assert domain == "legal"
        assert conf == 0.95

    def test_general_fallback(self):
        domain, conf = self.judge.classify("what is the weather like")
        assert domain == "general"
        assert conf == 0.90

    def test_priority_code_over_general(self):
        # 'code' keyword wins over later domains
        assert self.judge.classify("code about medical law")[0] == "code"


class TestJudgeFactory:
    def teardown_method(self):
        judge_factory.reset_judge()

    def test_deterministic_when_env_set(self, monkeypatch):
        monkeypatch.setenv("USE_DETERMINISTIC_INFERENCE", "true")
        judge = judge_factory.get_judge()
        assert isinstance(judge, DeterministicJudge)

    def test_singleton_cached(self, monkeypatch):
        monkeypatch.setenv("USE_DETERMINISTIC_INFERENCE", "true")
        assert judge_factory.get_judge() is judge_factory.get_judge()

    def test_reset_allows_new_instance(self, monkeypatch):
        monkeypatch.setenv("USE_DETERMINISTIC_INFERENCE", "true")
        first = judge_factory.get_judge()
        judge_factory.reset_judge()
        second = judge_factory.get_judge()
        assert first is not second

    def test_fallback_to_deterministic_on_import_failure(self, monkeypatch):
        monkeypatch.setenv("USE_DETERMINISTIC_INFERENCE", "false")
        import builtins
        real_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "app.judges.ollama_backend":
                raise ImportError("no ollama in CI")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=failing_import):
            judge = judge_factory.get_judge()
        assert isinstance(judge, DeterministicJudge)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _make_collector(interactions):
    collector = MagicMock()
    collector.get_all_interactions = AsyncMock(return_value=interactions)
    return collector


def _patched_collector(interactions):
    collector = _make_collector(interactions)
    return patch("app.rag.data_collector.get_data_collector", return_value=collector)


class TestCalibration:
    @pytest.fixture(autouse=True)
    def _tmp_calibration_file(self, tmp_path, monkeypatch):
        f = tmp_path / "confidence_calibration.json"
        monkeypatch.setattr(calibration, "CALIBRATION_FILE", f)
        yield

    @pytest.mark.asyncio
    async def test_insufficient_data_returns_default(self):
        interactions = [
            {"feedback_label": "incorrect", "confidence": 0.5}
        ] * 9
        with _patched_collector(interactions):
            threshold = await calibration.auto_calibrate_threshold()
        assert threshold == 0.75

    @pytest.mark.asyncio
    async def test_calibrates_to_95th_percentile(self):
        # 10 incorrect samples; 95th percentile of 0.0..0.9 is 0.855
        interactions = [
            {"feedback_label": "incorrect", "confidence": i / 10}
            for i in range(10)
        ]
        with _patched_collector(interactions):
            threshold = await calibration.auto_calibrate_threshold()
        assert abs(threshold - 0.855) < 1e-6
        data = json.loads(calibration.CALIBRATION_FILE.read_text())
        assert data["threshold"] == pytest.approx(threshold)
        assert data["sample_count"] == 10

    @pytest.mark.asyncio
    async def test_only_incorrect_feedback_counts(self):
        interactions = (
            [{"feedback_label": "correct", "confidence": 0.99}] * 20
            + [{"feedback_label": "incorrect", "confidence": i / 10} for i in range(10)]
        )
        with _patched_collector(interactions):
            threshold = await calibration.auto_calibrate_threshold()
        # Correct-labeled entries must not influence the percentile
        assert threshold < 0.99

    def test_get_threshold_default_without_file(self):
        assert calibration.get_confidence_threshold() == 0.75

    def test_get_threshold_from_file(self):
        calibration.CALIBRATION_FILE.write_text(json.dumps({"threshold": 0.61}))
        assert calibration.get_confidence_threshold() == 0.61


# ---------------------------------------------------------------------------
# SearchTool
# ---------------------------------------------------------------------------

class TestSearchTool:
    def _tool(self, client):
        tool = SearchTool.__new__(SearchTool)
        tool.client = client
        return tool

    def test_strict_search_not_found(self):
        client = MagicMock()
        client.search.return_value = []
        result = self._tool(client).strict_search("q")
        assert result["found"] is False
        assert result["results"] == []

    def test_strict_search_found(self):
        client = MagicMock()
        results = [{"url": "https://example.com/a"}, {"url": "https://example.com/b"}]
        client.search.return_value = results
        client.get_context.return_value = "context text"
        result = self._tool(client).strict_search("q")
        assert result["found"] is True
        assert result["top_url"] == "https://example.com/a"
        assert result["context"] == "context text"

    def test_get_search_tool_singleton(self):
        with patch("app.rag.search_tool.get_tavily_client", return_value=MagicMock()):
            a = get_search_tool()
            b = get_search_tool()
        assert a is b
