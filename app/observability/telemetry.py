import json
import os
import aiofiles
from datetime import datetime
from typing import Dict, Any

class TelemetryLogger:
    """
    Lightweight logger for audit telemetry, specifically for load testing correlation.
    """
    def __init__(self):
        # Ensure directory exists
        self.log_dir = "data"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "telemetry_audit.jsonl")

    async def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log a structured event to JSONL.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            **data
        }
        
        try:
            async with aiofiles.open(self.log_file, "a", encoding="utf-8") as f:
                await f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Telemetry logging failed: {e}")

# Global instance
_telemetry = None

def get_telemetry_logger() -> TelemetryLogger:
    global _telemetry
    if _telemetry is None:
        _telemetry = TelemetryLogger()
    return _telemetry
