"""
Structured logging coverage of new modules (audit follow-up).

app/kb, app/reasoners/factory, and error paths in app/main previously used
bare print(), which bypasses the request-ID-correlated structured logging
pipeline (see app/observability/logging_setup). These tests pin that:
1. no print() remains in the covered modules,
2. log records from those modules carry the request_id filter/formatter.
"""

import ast
import logging
from pathlib import Path

from app.observability.logging_setup import (
    JsonFormatter,
    RequestIdFilter,
    get_logger,
    request_id_var,
)

APP = Path(__file__).resolve().parents[1] / "app"

# Modules that shipped with print()-based diagnostics and were migrated to
# the structured logger. Keep this list in sync with new modules.
COVERED_MODULES = [
    APP / "kb" / "__init__.py",
    APP / "reasoners" / "factory.py",
]


class TestNoBarePrints:
    def test_covered_modules_have_no_print_calls(self):
        for path in COVERED_MODULES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            prints = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ]
            assert prints == [], f"{path.name} still has {len(prints)} print() calls"


class TestRequestCorrelation:
    def test_kb_logger_is_namespaced(self):
        logger = get_logger("kb")
        assert logger.name == "app.kb"
        # Root 'app' hierarchy carries the request-id filter via its handler.
        root = logging.getLogger("app")
        assert any(
            isinstance(f, RequestIdFilter)
            for h in root.handlers
            for f in h.filters
        )

    def test_record_carries_request_id(self):
        token = request_id_var.set("test-rid-123")
        try:
            logger = get_logger("kb")
            handler_filter = RequestIdFilter()
            record = logging.LogRecord(
                "app.kb", logging.WARNING, "p", 1, "boom %s", ("x",), None
            )
            assert handler_filter.filter(record) is True
            assert record.request_id == "test-rid-123"
        finally:
            request_id_var.reset(token)

    def test_json_formatter_includes_request_id(self):
        record = logging.LogRecord(
            "app.reasoners.factory", logging.INFO, "p", 1, "msg", None, None
        )
        record.request_id = "rid-42"
        import json as _json

        payload = _json.loads(JsonFormatter().format(record))
        assert payload["request_id"] == "rid-42"
        assert payload["logger"] == "app.reasoners.factory"
