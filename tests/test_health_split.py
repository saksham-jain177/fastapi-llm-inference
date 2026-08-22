"""
Tests for the readiness/liveness health endpoint split.

- /health/live: process-only liveness, never touches infra, always 200.
- /health/ready: infra probes; 503 only when a *configured* dependency is
  unreachable. Unconfigured deps (no REDIS_URL/MONGO_URL) don't gate.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_liveness_always_ok():
    """Liveness returns 200 with no infra checks."""
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readiness_ready_when_no_infra_configured(monkeypatch):
    """With no Redis/Mongo configured (CI/local), instance is ready."""
    from unittest.mock import MagicMock, patch

    collector = MagicMock()
    collector.redis_client = None
    collector.mongo_collection = None
    with patch("app.rag.data_collector.get_data_collector", return_value=collector):
        r = client.get("/health/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ready"


def test_readiness_503_when_configured_redis_down(monkeypatch):
    """Configured-but-unreachable Redis gates readiness with 503."""
    from unittest.mock import AsyncMock, MagicMock, patch

    collector = MagicMock()
    collector.redis_client = MagicMock()
    collector.redis_client.ping = AsyncMock(side_effect=ConnectionError("down"))
    collector.mongo_collection = None
    with patch("app.rag.data_collector.get_data_collector", return_value=collector):
        r = client.get("/health/ready")
    assert r.status_code == 503
    data = r.json()
    assert data["status"] == "unavailable"
    assert "redis" in data["unhealthy"]


def test_readiness_ready_when_all_probes_up():
    """All configured dependencies up -> ready."""
    from unittest.mock import AsyncMock, MagicMock, patch

    db = MagicMock()
    db.command = AsyncMock(return_value={"ok": 1})
    collector = MagicMock()
    collector.redis_client = MagicMock()
    collector.redis_client.ping = AsyncMock(return_value=True)
    collector.mongo_collection = MagicMock()
    collector.mongo_collection.database = db
    with patch("app.rag.data_collector.get_data_collector", return_value=collector):
        r = client.get("/health/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ready"
    assert data["redis"] == "up"
    assert data["mongo"] == "up"
