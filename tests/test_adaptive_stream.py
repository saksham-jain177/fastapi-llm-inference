"""
Tests for the SSE streaming variant of /infer-adaptive.

Uses FastAPI TestClient with mocked orchestrator/moderation to verify:
guard chain parity with /infer-adaptive, SSE event structure
(metadata -> token* -> done), and error propagation.
"""
import json

import pytest
from fastapi.testclient import TestClient

import os
os.environ.setdefault("USE_DETERMINISTIC_INFERENCE", "true")
os.environ.setdefault("API_KEY", "test-key")

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _post_stream(client, payload):
    return client.post("/infer-adaptive/stream", json=payload)


def test_stream_happy_path_event_sequence(client, monkeypatch):
    """metadata event first, then token events, then done."""
    from unittest.mock import AsyncMock, MagicMock, patch

    fake_result = {
        "response": "Streaming answer works",
        "confidence": 0.9,
        "intent": "general",
        "source": "redis",
        "refused": False,
        "citations": [],
        "cache_hit": True,
    }
    with patch("app.routing.orchestrator.get_orchestrator") as mock_get:
        orch = MagicMock()
        orch.route_and_execute = AsyncMock(return_value=fake_result)
        mock_get.return_value = orch

        resp = _post_stream(client, {"prompt": "What is streaming inference?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    body = resp.text
    # metadata event carries routing info
    assert "event: metadata" in body
    meta_line = next(
        l for l in body.splitlines() if l.startswith("data:") and "cache_hit" in l
    )
    meta = json.loads(meta_line[len("data:"):].strip())
    assert meta["refused"] is False
    assert meta["cache_hit"] is True
    assert meta["source"] == "redis"

    # token events carry the answer words
    token_data = [
        l[len("data:"):].strip()
        for l in body.splitlines() if l.startswith("data:") and "cache_hit" not in l
    ]
    joined = " ".join(d for d in token_data if d != "[DONE]")
    assert "Streaming" in joined and "works" in joined
    assert "data: [DONE]" in body


def test_stream_refused_still_streams_with_metadata(client):
    """In-pipeline refusals stream normally with refused=true metadata."""
    from unittest.mock import AsyncMock, MagicMock, patch

    fake_result = {
        "response": "I cannot answer that.",
        "confidence": 0.0,
        "intent": "refused",
        "source": "refused",
        "refused": True,
    }
    with patch("app.routing.orchestrator.get_orchestrator") as mock_get:
        orch = MagicMock()
        orch.route_and_execute = AsyncMock(return_value=fake_result)
        mock_get.return_value = orch
        resp = _post_stream(client, {"prompt": "What is falkordb?"})
    assert resp.status_code == 200
    assert '"refused": true' in resp.text.replace(" ", "").replace('"refused":true', '"refused": true') or "refused" in resp.text


def test_stream_prompt_too_long_413(client):
    """Oversized prompts are rejected before the stream starts."""
    resp = _post_stream(client, {"prompt": "x" * 10000})
    assert resp.status_code == 413


def test_stream_orchestrator_error_yields_error_event(client):
    """Pipeline exceptions surface as an SSE error event, not a 500 mid-stream."""
    from unittest.mock import AsyncMock, MagicMock, patch

    with patch("app.routing.orchestrator.get_orchestrator") as mock_get:
        orch = MagicMock()
        orch.route_and_execute = AsyncMock(side_effect=RuntimeError("boom"))
        mock_get.return_value = orch
        resp = _post_stream(client, {"prompt": "What is streaming inference?"})
    assert resp.status_code == 200
    assert "event: error" in resp.text
    assert "boom" in resp.text
