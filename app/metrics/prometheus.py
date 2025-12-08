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
    ['domain', 'method']  # method: semantic/llm_judge
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

# Confidence scores
classification_confidence = Histogram(
    'llm_classification_confidence',
    'Classification confidence scores',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
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
