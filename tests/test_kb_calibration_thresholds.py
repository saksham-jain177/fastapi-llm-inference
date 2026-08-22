"""
Regression tests for the KB-route × calibration-threshold audit findings.

1. auto_calibrate_threshold read `feedback_label`, but DataCollector stores
   labels under `feedback` — live data was never seen and the threshold was
   silently pinned to 0.75. Both keys must now be honored.
2. Negative labels are the 3-label RLHF schema: incorrect AND
   should_have_refused both count (parity with scripts/calibration_report).
3. Suggested threshold is clamped to [0.30, 0.95] like the offline report.
4. The calibration report surfaces the current KB evidence threshold
   (KB_MIN_SIMILARITY) as context, without suggesting changes to it.
"""

import json

import pytest

from app.models import calibration


@pytest.fixture()
def tmp_calibration_file(tmp_path, monkeypatch):
    f = tmp_path / "confidence_calibration.json"
    monkeypatch.setattr(calibration, "CALIBRATION_FILE", f)
    return f


def _patched_collector(interactions):
    from unittest.mock import AsyncMock, MagicMock, patch

    collector = MagicMock()
    collector.get_all_interactions = AsyncMock(return_value=interactions)
    return patch(
        "app.rag.data_collector.get_data_collector", return_value=collector
    )


class TestLiveFeedbackKey:
    @pytest.mark.asyncio
    async def test_feedback_key_is_read(self, tmp_calibration_file):
        """Regression: labels stored under 'feedback' (live schema) calibrate."""
        interactions = [
            {"feedback": "incorrect", "confidence": i / 10} for i in range(10)
        ]
        with _patched_collector(interactions):
            threshold = await calibration.auto_calibrate_threshold()
        assert threshold != 0.75  # would be pinned default if key unreadable
        assert threshold == pytest.approx(0.855)

    @pytest.mark.asyncio
    async def test_legacy_feedback_label_still_read(self, tmp_calibration_file):
        interactions = [
            {"feedback_label": "incorrect", "confidence": i / 10} for i in range(10)
        ]
        with _patched_collector(interactions):
            threshold = await calibration.auto_calibrate_threshold()
        assert threshold == pytest.approx(0.855)

    @pytest.mark.asyncio
    async def test_should_have_refused_counts_as_negative(self, tmp_calibration_file):
        interactions = [{"feedback": "should_have_refused", "confidence": 0.9}] * 10
        with _patched_collector(interactions):
            threshold = await calibration.auto_calibrate_threshold()
        # 10 identical samples -> p95 = 0.9, within clamp band
        assert threshold == pytest.approx(0.9)


class TestClampBand:
    @pytest.mark.asyncio
    async def test_high_outlier_sample_clamped(self, tmp_calibration_file):
        interactions = [{"feedback": "incorrect", "confidence": 1.0} for _ in range(10)]
        with _patched_collector(interactions):
            threshold = await calibration.auto_calibrate_threshold()
        assert threshold == pytest.approx(0.95)  # clamped from 1.0

    @pytest.mark.asyncio
    async def test_low_sample_clamped(self, tmp_calibration_file):
        interactions = [{"feedback": "incorrect", "confidence": 0.0} for _ in range(10)]
        with _patched_collector(interactions):
            threshold = await calibration.auto_calibrate_threshold()
        assert threshold == pytest.approx(0.30)  # clamped up from 0.0

    def test_persisted_value_matches_return(self, tmp_calibration_file):
        interactions = [{"feedback": "incorrect", "confidence": 0.5} for _ in range(12)]
        import asyncio

        with _patched_collector(interactions):
            threshold = asyncio.run(calibration.auto_calibrate_threshold())
        saved = json.loads(tmp_calibration_file.read_text())
        assert saved["threshold"] == pytest.approx(threshold)
        assert saved["sample_count"] == 12


class TestReportSurfacesKbThreshold:
    def test_compute_suggestions_includes_kb_min_similarity(self):
        from scripts.calibration_report import compute_suggestions

        result = compute_suggestions([], min_samples=10)
        assert "current_kb_min_similarity" in result
        assert 0.0 <= result["current_kb_min_similarity"] <= 1.0

    def test_render_shows_kb_threshold_as_context_only(self):
        from scripts.calibration_report import compute_suggestions, render

        rendered = render(compute_suggestions([], min_samples=10))
        assert "KB min similarity" in rendered
        assert "not auto-suggested" in rendered
