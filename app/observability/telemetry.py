import json
import os
import asyncio
from datetime import datetime
from typing import Dict, Any


class TelemetryLogger:
    """
    Lightweight logger for audit telemetry, specifically for load testing correlation.
    Uses synchronous file append (safe for append-only JSONL, avoids aiofiles dependency).
    """
    def __init__(self):
        # Ensure directory exists
        self.log_dir = "data"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "telemetry_audit.jsonl")

    async def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log a structured event to JSONL.
        Uses sync append in executor to avoid blocking the event loop.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            **data
        }

        line = json.dumps(entry, ensure_ascii=False) + "\n"

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._write_line, line)
        except Exception as e:
            print(f"Telemetry logging failed: {e}")

    def _write_line(self, line: str):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line)


# Global instance
_telemetry = None


def get_telemetry_logger() -> TelemetryLogger:
    global _telemetry
    if _telemetry is None:
        _telemetry = TelemetryLogger()
    return _telemetry
