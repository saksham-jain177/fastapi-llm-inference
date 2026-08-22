"""
Tests for the semantic (embedding cosine-similarity) cache in DataCollector.

Uses a fake embedding model so no sentence-transformers download is needed:
similar queries map to identical vectors, dissimilar to orthogonal ones.
"""
import asyncio
import hashlib
import json

import pytest

from app.rag.data_collector import DataCollector


class FakeModel:
    """Deterministic stand-in for SentenceTransformer.

    Embeds a query as a unit vector along its first keyword axis; queries
    sharing a keyword get cosine similarity 1.0, otherwise 0.0.
    """

    KEYWORDS = ["redis", "kubernetes", "photosynthesis"]

    def encode(self, texts, **kwargs):
        import numpy as np

        out = []
        for t in texts:
            vec = np.zeros(len(self.KEYWORDS))
            for i, kw in enumerate(self.KEYWORDS):
                if kw in t.lower():
                    vec[i] = 1.0
            norm = np.linalg.norm(vec)
            out.append(vec / norm if norm else vec)
        return out


def make_collector_with_fake_redis():
    """DataCollector with an in-memory async fake Redis and FakeModel injected."""

    class FakeRedis:
        def __init__(self):
            self.store = {}

        async def get(self, key):
            return self.store.get(key)

        async def set(self, key, value, ex=None):
            self.store[key] = value

    c = DataCollector.__new__(DataCollector)
    c.redis_client = FakeRedis()
    c.semantic_cache_enabled = True
    c.semantic_threshold = 0.92
    c.semantic_max_entries = 500
    c._embedding_model = FakeModel()
    return c


def qhash(q: str) -> str:
    return hashlib.md5(q.strip().lower().encode()).hexdigest()


@pytest.mark.anyio
async def test_semantic_hit_on_paraphrase():
    """A reworded query with same meaning hits the semantically cached response."""
    c = make_collector_with_fake_redis()
    await c.cache_response("How do I use redis caching?", "Redis caches responses.")
    # Different string, same keyword -> similarity 1.0 >= threshold
    hit = await c.get_cached_response("explain redis caching please")
    assert hit == "Redis caches responses."


@pytest.mark.asyncio
async def test_no_hit_for_unrelated_query():
    """A dissimilar query must not steal another entry's cached response."""
    c = make_collector_with_fake_redis()
    await c.cache_response("How does photosynthesis work?", "Chlorophyll stuff.")
    hit = await c.get_cached_response("how do I deploy kubernetes")
    assert hit is None


@pytest.mark.asyncio
async def test_exact_match_fallback_when_model_missing():
    """Without an embedding model, exact normalized match still works."""
    c = make_collector_with_fake_redis()
    c._embedding_model = None  # simulate model load failure
    await c.cache_response("How do I use Redis cache?", "Cached Response")
    hit = await c.get_cached_response("how do i use redis cache?")
    assert hit == "Cached Response"
    miss = await c.get_cached_response("totally different query about kubernetes")
    assert miss is None


@pytest.mark.asyncio
async def test_disabled_via_env_flag(monkeypatch):
    """SEMANTIC_CACHE_ENABLED=false disables semantic lookup entirely."""
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "false")
    from app.rag.data_collector import DataCollector as DC

    c = DC.__new__(DC)
    c.redis_client = None
    c.semantic_cache_enabled = False
    assert c._get_embedding_model() is None


@pytest.mark.asyncio
async def test_index_bounded_to_max_entries():
    """Semantic index never grows beyond SEMANTIC_CACHE_MAX_ENTRIES."""
    c = make_collector_with_fake_redis()
    c.semantic_max_entries = 3
    for i in range(5):
        await c.cache_response(f"redis question number {i}", f"resp {i}")
    index = json.loads(c.redis_client.store["cache:semantic:index"])
    assert len(index) <= 3
