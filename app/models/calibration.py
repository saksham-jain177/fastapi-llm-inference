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
    
    # Extract confidence scores from incorrect answers
    incorrect_confidences = []
    for entry in interactions:
        if entry.get("feedback_label") == "incorrect" and "confidence" in entry:
            incorrect_confidences.append(entry["confidence"])
    
    # Bootstrap: need minimum data
    if len(incorrect_confidences) < 10:
        print(f"  Insufficient data for calibration ({len(incorrect_confidences)}/10). Using default 0.75")
        return 0.75
    
    # Compute 95th percentile
    threshold = float(np.percentile(incorrect_confidences, 95))
    
    print(f"  Calibrated threshold: {threshold:.3f} (based on {len(incorrect_confidences)} incorrect samples)")
    
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
