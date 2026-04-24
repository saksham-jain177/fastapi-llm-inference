import sys
import contextvars
from loguru import logger  # type: ignore
from types import FrameType
from typing import cast

# Context variable to store the request ID
request_id_cvar: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

def _request_id_filter(record):
    """Filter to inject request_id into every log record."""
    record["extra"]["request_id"] = request_id_cvar.get()
    return True

# Remove the default logger
logger.remove()

# Add a standard JSON logger for production
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[request_id]} | {name}:{function}:{line} - {message}",
    filter=_request_id_filter,
    level="INFO",
    serialize=False, # Set to True in actual prod for raw JSON, False is better for local dev reading. We'll use False for better human readability in terminal, but with structured tags.
    enqueue=True,    # Thread-safe async logging
    diagnose=False,  # SECURITY: Prevent leaking local variables (like user prompts) in exception traces
    backtrace=False, # SECURITY: Prevent verbose tracing which can leak internal PII state
)

def get_logger():
    return logger
