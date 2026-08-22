"""
Per-route latency histograms for the truth-first routing pipeline.

Adds `orchestrator_route_latency_seconds` with a `route` label covering every
routing path (refused, redis_cache, adapter, rag-external, model, refused_*),
plus `orchestrator_route_total` for exact route counts.

Instrumentation lives in the Orchestrator itself (not the endpoints) so every
caller — /infer-adaptive, /infer-adaptive/stream, future endpoints — is
covered by construction, and the measured window is the pipeline, not the
HTTP layer (which already has llm_request_latency_seconds).
"""

import time
from functools import wraps

from prometheus_client import Counter, Histogram

route_latency = Histogram(
    'orchestrator_route_latency_seconds',
    'Orchestrator pipeline latency by routing path',
    ['route'],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

route_total = Counter(
    'orchestrator_route_total',
    'Orchestrator requests by final routing path',
    ['route'],
)

# Response-data key that carries the routing path in every branch of
# Orchestrator.route_and_execute.
_ROUTE_KEY = "mode"


def _normalize_route(response_data: dict) -> str:
    route = response_data.get(_ROUTE_KEY) or "unknown"
    return str(route)


def instrument_orchestrator(cls):
    """
    Class decorator: wraps Orchestrator.route_and_execute with per-route
    latency/counter metrics derived from the result's `mode`.

    Failure path: if the pipeline raises, the sample is recorded under
    route="error" so error latency is visible too.
    """
    original = cls.route_and_execute

    @wraps(original)
    async def wrapped(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            result = await original(self, *args, **kwargs)
        except Exception:
            route_latency.labels(route="error").observe(time.perf_counter() - start)
            route_total.labels(route="error").inc()
            raise
        elapsed = time.perf_counter() - start
        route = _normalize_route(result)
        route_latency.labels(route=route).observe(elapsed)
        route_total.labels(route=route).inc()
        return result

    cls.route_and_execute = wrapped
    return cls
