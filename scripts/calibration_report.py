"""
Feedback-driven gate calibration (offline, read-only).

Analyzes accumulated RLHF feedback labels in Mongo (via the DataCollector) and
computes SUGGESTED epistemic/refusal thresholds. Prints current vs suggested
values side by side.

This tool NEVER writes: it does not touch data/confidence_calibration.json,
Mongo, or any runtime state. Applying a suggestion is a deliberate human
decision (edit the calibration file / env config yourself).

Usage:
    uv run python -m scripts.calibration_report [options]

Options:
    --min-samples N     Minimum labeled samples required per estimate (default 10)
    --json              Machine-readable output
Exit codes:
    0 = suggestions computed, 1 = insufficient data (suggestions are None)
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from typing import Optional

DEFAULT_THRESHOLD = 0.75  # mirrors app/models/calibration.py bootstrap default


def _percentile(sorted_vals, pct):
    """Linear-interpolated percentile on a pre-sorted list (no numpy needed)."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    frac = k - int(k)
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def compute_suggestions(
    interactions: list,
    min_samples: int = 10,
) -> dict:
    """
    Pure computation over interaction dicts (testable, no I/O).

    Logic (aligned with app/models/calibration.py's auto-calibrate approach):

    - epistemic_threshold: the gate should refuse answers humans later marked
      incorrect/should_have_refused. Suggested threshold = 95th percentile of
      confidences among those negative labels — i.e. set the bar just above
      the confidence level at which most mistakes happen.
    - semantic_threshold: kept informational here; feedback does not carry
      semantic scores, so we only report label coverage for it.

    The KB evidence threshold (KB_MIN_SIMILARITY) is a separate knob with a
    different unit (cosine-derived embedding similarity, NOT model
    confidence): it is reported for context but never suggested here —
    changing it changes what counts as EVIDENCE, which must stay a deliberate,
    human-reviewed decision. See docs/internal_kb.md "Threshold relationships".
    """
    negative_confidences = []   # incorrect + should_have_refused
    correct_confidences = []
    label_counts = Counter()

    for entry in interactions:
        label = entry.get("feedback")
        if not label:
            continue
        label_counts[label] += 1
        conf = entry.get("confidence")
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            continue
        if label in ("incorrect", "should_have_refused"):
            negative_confidences.append(conf)
        elif label == "correct":
            correct_confidences.append(conf)

    negative_confidences.sort()
    correct_confidences.sort()

    result = {
        "labeled_total": sum(label_counts.values()),
        "label_counts": dict(label_counts),
        "negative_count": len(negative_confidences),
        "correct_count": len(correct_confidences),
        "current_epistemic_threshold": _load_current_threshold(),
        "current_kb_min_similarity": _load_kb_min_similarity(),
        "suggested_epistemic_threshold": None,
        "basis": None,
        "correct_above_suggested": None,
        "false_refusal_risk_pct": None,
    }

    if len(negative_confidences) < min_samples:
        result["basis"] = (
            f"insufficient data: {len(negative_confidences)} negative samples "
            f"(need >= {min_samples})"
        )
        return result

    p95 = round(_percentile(negative_confidences, 95), 4)

    # Clamp to a sane band around the default so a tiny weird sample set can't
    # propose an always-refuse (>=1.0) or never-gate (<0.3) threshold.
    suggested = max(0.30, min(0.95, p95))

    # Cost check: how many CORRECT answers would this threshold have refused?
    above = sum(1 for c in correct_confidences if c < suggested)
    false_refusal_pct = (
        round(100.0 * above / len(correct_confidences), 2) if correct_confidences else None
    )

    result.update(
        {
            "suggested_epistemic_threshold": suggested,
            "basis": f"95th percentile of {len(negative_confidences)} "
            "negative-label confidences (clamped to [0.30, 0.95])",
            "correct_below_suggested": above,
            "false_refusal_risk_pct": false_refusal_pct,
        }
    )
    return result


def _load_current_threshold() -> float:
    """Read the currently-applied epistemic threshold (file, else default)."""
    from app.models.calibration import get_confidence_threshold

    return get_confidence_threshold()


def _load_kb_min_similarity() -> float:
    """Read the KB evidence threshold (reported for context; never suggested)."""
    from app.kb import min_similarity

    return min_similarity()


async def _gather_interactions() -> list:
    from app.rag.data_collector import get_data_collector

    collector = get_data_collector()
    return await collector.get_all_interactions()


def render(result: dict, as_json: bool = False) -> str:
    if as_json:
        return json.dumps(result, indent=2)

    lines = [
        "=" * 62,
        " Gate calibration report (offline — nothing is auto-applied)",
        "=" * 62,
        f" Labeled feedback samples : {result['labeled_total']}",
        f"   by label               : {result['label_counts']}",
        f" Current epistemic thresh.: {result['current_epistemic_threshold']}",
        f" Current KB min similarity: {result['current_kb_min_similarity']}"
        "  (evidence gate — separate knob, not auto-suggested)",
    ]
    if result["suggested_epistemic_threshold"] is None:
        lines.append(f" Suggested threshold      : N/A ({result['basis']})")
    else:
        lines += [
            f" Suggested threshold      : {result['suggested_epistemic_threshold']}"
            f"  ({result['basis']})",
            f" Correct answers that would now be gated: "
            f"{result['correct_below_suggested']} "
            f"({result['false_refusal_risk_pct']}% false-refusal risk)",
            "",
            " To apply manually: update data/confidence_calibration.json "
            "(or KnowledgeGate init) after reviewing the risk figure above.",
        ]
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    interactions = asyncio.run(_gather_interactions())
    result = compute_suggestions(interactions, min_samples=args.min_samples)
    print(render(result, as_json=args.json))
    return 0 if result["suggested_epistemic_threshold"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
