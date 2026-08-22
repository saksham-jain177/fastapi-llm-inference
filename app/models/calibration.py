"""
Auto-calibrating confidence threshold.
Prevents drift by recomputing based on labeled feedback data.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional


CALIBRATION_FILE = Path("data/confidence_calibration.json")


async def auto_calibrate_threshold() -> float:
    """
    Recompute threshold as 95th percentile of incorrect answers' confidence.
    Run daily via cron or on app startup.
    
    Returns:
        Calibrated threshold value
    """
    from app.rag.data_collector import get_data_collector
    
    collector = get_data_collector()
    interactions = await collector.get_all_interactions()
    
    # Extract confidence scores from negative-label answers
    # ("incorrect" and "should_have_refused" — the 3-label RLHF schema).
    # NOTE: feedback is stored under the "feedback" key by DataCollector.
    # The legacy "feedback_label" key is accepted for backwards compatibility
    # with older records, but reading ONLY it means live data is never seen
    # and calibration silently pins to the 0.75 default (audit finding).
    incorrect_confidences = []
    for entry in interactions:
        label = entry.get("feedback") or entry.get("feedback_label")
        if label == "incorrect" and entry.get("confidence") is not None:
            incorrect_confidences.append(entry["confidence"])
        elif label == "should_have_refused" and entry.get("confidence") is not None:
            # Refused-after-the-fact means the answer was overconfident;
            # include it so the threshold covers that failure mode too,
            # matching scripts/calibration_report.compute_suggestions.
            incorrect_confidences.append(entry["confidence"])
    
    # Clamp to the same sane band as the offline report so a tiny weird
    # sample can't propose an always-refuse (>=1.0) or never-gate (<0.3) bar.
    CLAMP_MIN, CLAMP_MAX = 0.30, 0.95

    # Bootstrap: need minimum data
    if len(incorrect_confidences) < 10:
        print(f"  Insufficient data for calibration ({len(incorrect_confidences)}/10). Using default 0.75")
        return 0.75
    
    # Compute 95th percentile (clamped)
    threshold = float(np.percentile(incorrect_confidences, 95))
    threshold = max(CLAMP_MIN, min(CLAMP_MAX, threshold))
    
    print(f"  Calibrated threshold: {threshold:.3f} (based on {len(incorrect_confidences)} negative samples)")
    
    # Save for persistence
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump({
            "threshold": threshold,
            "updated_at": __import__('time').time(),
            "sample_count": len(incorrect_confidences)
        }, f)
    
    return threshold


def get_confidence_threshold() -> float:
    """
    Load current calibrated threshold.
    Falls back to 0.75 if no calibration file exists.
    """
    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE) as f:
            data = json.load(f)
            return data["threshold"]
    
    return 0.75  # Bootstrap default
