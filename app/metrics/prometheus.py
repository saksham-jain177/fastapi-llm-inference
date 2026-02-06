"""
Prometheus metrics collection for LLM inference API.
Tracks requests, latency, cache hits, adapter usage.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from functools import wraps
import time


# Request counters
requests_total = Counter(
    'llm_requests_total',
    'Total number of requests',
    ['endpoint', 'status']
)

# Latency histogram
request_latency = Histogram(
    'llm_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint']
)

# Active requests
active_requests = Gauge(
    'llm_active_requests',
    'Number of currently active requests'
)

# Domain classification
domain_classifications = Counter(
    'llm_domain_classifications_total',
    'Domain classification counts',
    ['domain', 'method']  # method: semantic/hardware_independent_judge
)

# Adapter usage
adapter_usage = Counter(
    'llm_adapter_usage_total',
    'Adapter usage counts',
    ['domain']
)

# Cache metrics
cache_hits = Counter(
    'llm_cache_hits_total',
    'Cache hit counts',
    ['cache_type']  # tavily/embeddings
)

cache_misses = Counter(
    'llm_cache_misses_total',
    'Cache miss counts',
    ['cache_type']
)

# Infrastructure Metrics
redis_cache_hit_total = Counter(
    'redis_cache_hit_total',
    'Total number of successful Redis cache hits'
)

redis_cache_miss_total = Counter(
    'redis_cache_miss_total',
    'Total number of Redis cache misses'
)

mongodb_read_total = Counter(
    'mongodb_read_total',
    'Total number of MongoDB read operations'
)

mongodb_write_total = Counter(
    'mongodb_write_total',
    'Total number of MongoDB write operations'
)


# Epistemic Gating Metrics
gate_decision_total = Counter(
    'gate_decision_total',
    'Total count of epistemic gate decisions',
    ['decision', 'reason'] # allow/high_conf, fallback_rag/low_conf_novel, refuse/low_conf_stale
)

hallucination_counter = Counter(
    'hallucination_counter_total',
    'Count of detected potential hallucinations'
)

refusal_counter = Counter(
    'refusal_counter_total',
    'Count of model refusals / abstentions'
)

# Semantic API Contract Metrics
response_refusal_total = Counter(
    'response_refusal_total',
    'Total number of refused responses exposed to the client'
)

response_confidence_bucket_total = Counter(
    'response_confidence_bucket_total',
    'Total number of responses by confidence bucket',
    ['bucket'] # low, medium, high
)


# Inference specific latency (distinct from request latency)
inference_latency = Histogram(
    'inference_latency_seconds',
    'Internal model inference latency',
    ['mode'] # confident, search, fallback
)

# Confidence scores
classification_confidence = Histogram(
    'llm_classification_confidence',
    'Confidence scores of memory short-circuit hits (avg derived)',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

# RAG Fallback Metrics
rag_tavily_fallback_total = Counter(
    "rag_tavily_fallback_total",
    "Total times Tavily failed and fallback was attempted"
)

rag_duckduckgo_used_total = Counter(
    "rag_duckduckgo_used_total",
    "Total times DuckDuckGo was successfully used as fallback"
)


def track_request(endpoint: str):
    """
    Decorator to track request metrics.
    
    Usage:
        @track_request("infer")
        def my_endpoint():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            active_requests.inc()
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                requests_total.labels(endpoint=endpoint, status=status).inc()
                request_latency.labels(endpoint=endpoint).observe(duration)
                active_requests.dec()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            active_requests.inc()
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                requests_total.labels(endpoint=endpoint, status=status).inc()
                request_latency.labels(endpoint=endpoint).observe(duration)
                active_requests.dec()
        
        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def get_metrics():
    """
    Get current metrics in Prometheus format.
    
    Returns:
        Tuple of (metrics_text, content_type)
    """
    return generate_latest(), CONTENT_TYPE_LATEST


def get_system_stats() -> dict:
    """
    Get simplified internal stats for the frontend dashboard.
    """
    stats = {
        "active_requests": active_requests._value.get(),
        "total_requests": sum(c._value.get() for c in requests_total.collect()[0].samples),
        "total_errors": sum(s.value for s in requests_total.collect()[0].samples if s.labels['status'] == 'error'),
        "cache_hits": sum(c._value.get() for c in cache_hits.collect()[0].samples) if cache_hits.collect() else 0,
        "cache_misses": sum(c._value.get() for c in cache_misses.collect()[0].samples) if cache_misses.collect() else 0,
        "adapter_usage": {
             s.labels['domain']: s.value for s in adapter_usage.collect()[0].samples
        } if adapter_usage.collect() else {}
    }
    return stats
