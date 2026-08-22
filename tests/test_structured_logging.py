"""
Tests for structured logging + request-ID correlation.

Covers: middleware assignment/propagation of X-Request-ID, contextvar
binding visible to log records, and JSON formatter output.
"""
import logging

import pytest
from fastapi.testclient import TestClient

import os
os.environ.setdefault("USE_DETERMINISTIC_INFERENCE", "true")

from app.main import app
from app.observability.logging_setup import (
    JsonFormatter,
    RequestIdFilter,
    get_logger,
    new_request_id,
    request_id_var,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_middleware_assigns_request_id(client):
    """Every response carries an X-Request-ID header; two requests differ."""
    r1 = client.get("/health")
    r2 = client.get("/health")
    rid1 = r1.headers.get("X-Request-ID")
    rid2 = r2.headers.get("X-Request-ID")
    assert rid1 and rid2
    assert rid1 != rid2


def test_middleware_propagates_incoming_request_id(client):
    """A caller-supplied X-Request-ID is echoed back unchanged."""
    r = client.get("/health", headers={"X-Request-ID": "my-trace-123"})
    assert r.headers["X-Request-ID"] == "my-trace-123"


def test_contextvar_bound_to_log_records():
    """LogRecords emitted inside a request context carry its request_id."""
    logger = get_logger("test.rid")
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    handler.addFilter(RequestIdFilter())
    logger.addHandler(handler)
    logger.propagate = False

    token = request_id_var.set("abc-42")
    try:
        logger.info("hello")
        logger.info("world")
    finally:
        request_id_var.reset(token)

    logger.removeHandler(handler)
    assert [r.request_id for r in records] == ["abc-42", "abc-42"]

    # Outside a request context the filter falls back to "-"
    rec = logging.LogRecord("x", logging.INFO, "p", 1, "m", None, None)
    assert RequestIdFilter().filter(rec) is True
    assert rec.request_id == "-"


def test_new_request_id_unique_and_short():
    ids = {new_request_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(len(i) == 12 for i in ids)


def test_json_formatter_output():
    rec = logging.LogRecord("app.test", logging.INFO, "p", 1, "msg here", None, None)
    rec.request_id = "rid-7"
    import json
    payload = json.loads(JsonFormatter().format(rec))
    assert payload["msg"] == "msg here"
    assert payload["request_id"] == "rid-7"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
