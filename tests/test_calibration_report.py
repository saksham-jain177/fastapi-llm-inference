"""Tests for the feedback-driven gate calibration report (offline, pure logic)."""

import os

os.environ["USE_DETERMINISTIC_INFERENCE"] = "true"

import pytest

from scripts.calibration_report import compute_suggestions, render


def _entry(label, conf):
    return {"feedback": label, "confidence": conf}


class TestComputeSuggestions:
    def test_insufficient_data_yields_no_suggestion(self):
        data = [_entry("incorrect", 0.5) for _ in range(5)]
        result = compute_suggestions(data, min_samples=10)
        assert result["suggested_epistemic_threshold"] is None
        assert "insufficient" in result["basis"]
        assert result["negative_count"] == 5

    def test_unlabeled_interactions_ignored(self):
        data = [{"feedback": None, "confidence": 0.9}] * 20
        result = compute_suggestions(data, min_samples=10)
        assert result["labeled_total"] == 0
        assert result["suggested_epistemic_threshold"] is None

    def test_invalid_confidence_skipped(self):
        data = [_entry("incorrect", "not-a-number") for _ in range(15)]
        result = compute_suggestions(data, min_samples=10)
        assert result["suggested_epistemic_threshold"] is None
        assert result["negative_count"] == 0

    def test_p95_of_negatives_becomes_suggestion(self):
        # 20 negatives spread 0.4..0.9; p95 should be ~0.875
        neg = [_entry("should_have_refused", 0.4 + i * 0.5 / 19) for i in range(20)]
        pos = [_entry("correct", 0.95)] * 10
        result = compute_suggestions(neg + pos, min_samples=10)
        assert result["negative_count"] == 20
        sugg = result["suggested_epistemic_threshold"]
        assert 0.85 <= sugg <= 0.95  # just above the mistake band
        assert result["correct_below_suggested"] == 0  # correct@0.95 stays allowed
        assert result["false_refusal_risk_pct"] == 0.0

    def test_clamped_to_sane_band(self):
        # All mistakes at confidence 1.0 → raw p95=1.0 must clamp to 0.95
        neg = [_entry("incorrect", 1.0) for _ in range(15)]
        result = compute_suggestions(neg, min_samples=10)
        assert result["suggested_epistemic_threshold"] == 0.95

    def test_floor_clamp(self):
        neg = [_entry("incorrect", 0.05) for _ in range(15)]
        result = compute_suggestions(neg, min_samples=10)
        assert result["suggested_epistemic_threshold"] == 0.30

    def test_false_refusal_risk_reported(self):
        neg = [_entry("incorrect", 0.7)] * 12
        pos = [_entry("correct", 0.6)] * 3 + [_entry("correct", 0.9)] * 9
        result = compute_suggestions(neg + pos, min_samples=10)
        # suggested ≈ clamp(p95 of [0.7]*12)=0.70; correct below: the three 0.6s
        assert result["correct_below_suggested"] == 3
        assert result["false_refusal_risk_pct"] == 25.0

    def test_current_threshold_loaded(self):
        result = compute_suggestions([], min_samples=10)
        assert result["current_epistemic_threshold"] == pytest.approx(0.75)


class TestRender:
    def test_json_mode_is_valid_json(self):
        result = compute_suggestions([], min_samples=1)
        import json

        parsed = json.loads(render(result, as_json=True))
        assert parsed["label_counts"] == {}

    def test_human_render_mentions_no_auto_apply(self):
        neg = [_entry("incorrect", 0.7)] * 12
        text = render(compute_suggestions(neg))
        assert "nothing is auto-applied" in text

    def test_insufficient_render_shows_basis(self):
        text = render(compute_suggestions([], min_samples=10))
        assert "N/A" in text and "insufficient" in text
