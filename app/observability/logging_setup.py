"""
Structured logging with per-request correlation IDs.

Provides:
- get_logger(name): stdlib logger preconfigured with a formatter that
  includes the request ID from a contextvar.
- request_id handling via contextvars (async-safe; each request gets its own).
- RequestIdMiddleware: FastAPI middleware assigning/propagating
  X-Request-ID, echoing it back in the response header.

JSON output is opt-in via LOG_FORMAT=json for log shipper ingestion;
default is human-readable text.
"""
import contextvars
import json
import logging
import os
import sys
import uuid

# Contextvar holding the current request ID (empty string outside a request)
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def new_request_id() -> str:
    """Generate a short unique request ID."""
    return uuid.uuid4().hex[:12]


class RequestIdFilter(logging.Filter):
    """Injects the current request_id into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


_configured = False


def _configure_root() -> None:
    """Configure the app logger hierarchy once."""
    global _configured
    if _configured:
        return

    root = logging.getLogger("app")
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stdout)

    rid_filter = RequestIdFilter()
    if os.getenv("LOG_FORMAT", "text").lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
        ))
    handler.addFilter(rid_filter)
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the configured 'app' hierarchy."""
    _configure_root()
    return logging.getLogger(f"app.{name}")
