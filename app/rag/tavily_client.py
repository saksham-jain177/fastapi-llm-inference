"""
Tavily RAG client with caching, rate limiting, and error handling.
"""

import os
import time
import asyncio
import hashlib
from tavily import TavilyClient
from typing import List, Dict, Optional
from functools import lru_cache
from datetime import datetime, timedelta
from app.observability.logger import get_logger

logger = get_logger()


class RateLimiter:
    """Redis-backed rate limiter with memory fallback for API calls."""
    
    def __init__(self, max_calls: int = 10, period_seconds: int = 60):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.local_calls = []
    
    async def wait_if_needed(self):
        """Asynchronously wait if rate limit is exceeded using Redis sliding window."""
        import uuid
        now = time.time()
        
        # Try Redis first
        try:
            from app.rag.data_collector import get_data_collector
            collector = get_data_collector()
            if collector.redis_client:
                key = "rate_limit:tavily_api"
                window_start = now - self.period_seconds
                
                # Cleanup old calls
                await collector.redis_client.zremrangebyscore(key, 0, window_start)
                
                # Count current
                count = await getattr(collector.redis_client, "zcard")(key)
                
                if count >= self.max_calls:
                    oldest = await getattr(collector.redis_client, "zrange")(key, 0, 0, withscores=True)
                    if oldest:
                        oldest_time = oldest[0][1]
                        wait_time = max(0.0, self.period_seconds - (now - oldest_time))
                        logger.warning("Redis rate limit reached", wait_time=f"{wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                        # Re-calculate now after wait
                        now = time.time()
                
                # Add this request
                await getattr(collector.redis_client, "zadd")(key, {str(uuid.uuid4()): now})
                await collector.redis_client.expire(key, self.period_seconds * 2)
                return
        except Exception as e:
            logger.error("Redis rate limiter failed, falling back to local memory", error=str(e))
            
        # Fallback Local memory sliding window
        self.local_calls = [c for c in self.local_calls if now - c < self.period_seconds]
        if len(self.local_calls) >= self.max_calls:
            oldest_call = min(self.local_calls)
            wait_time = max(0.0, self.period_seconds - (now - oldest_call))
            logger.warning("Local memory rate limit reached", wait_time=f"{wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            now = time.time() # Update now after sleep
                
        self.local_calls.append(now)


class CachedTavilyRAG:
    """
    Tavily client with guardrails:
    - In-memory LRU cache (128 queries)
    - Rate limiting (10 requests/minute by default)
    - Retry logic with exponential backoff
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_requests_per_minute: int = 10,
        cache_size: int = 128
    ):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")
        
        self.client = TavilyClient(api_key=self.api_key)
        self.rate_limiter = RateLimiter(max_calls=max_requests_per_minute, period_seconds=60)
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_cache_key(self, query: str, max_results: int) -> str:
        """Generate cache key from query parameters."""
        content = f"{query}:{max_results}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _add_to_cache(self, key: str, value: List[Dict]):
        """Add to cache with LRU eviction."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO, not true LRU but sufficient)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
    
    async def search(self, query: str, max_results: int = 3, max_retries: int = 3) -> List[Dict]:
        """
        Search with caching and rate limiting.
        
        Args:
            query: Search query
            max_results: Maximum results
            max_retries: Maximum retry attempts
            
        Returns:
            List of search results
        """
        cache_key = self._get_cache_key(query, max_results)
        
        # Check cache
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        
        # Rate limiting wait
        await self.rate_limiter.wait_if_needed()
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.search(
                        query=query,
                        search_depth="advanced",
                        max_results=max_results
                    )
                )
                
                results = []
                for result in response.get("results", []):
                    results.append({
                        "title": result.get("title", ""),
                        "content": result.get("content", ""),
                        "url": result.get("url", ""),
                        "score": result.get("score", 0.0)
                    })
                
                # Cache successful results
                self._add_to_cache(cache_key, results)
                return results
            
            except Exception as e:
                if attempt < max_retries - 1:
                    backoff = 2 ** attempt
                    logger.warning("Tavily API error, retrying", error=str(e), attempt=attempt+1, max_retries=max_retries, backoff=f"{backoff}s")
                    await asyncio.sleep(backoff)
                else:
                    logger.error("Tavily API error after max attempts", error=str(e), max_retries=max_retries)
                    return []
    
    async def get_context(self, query: str, max_results: int = 3) -> str:
        """Get formatted context string from search results."""
        results = await self.search(query, max_results)
        
        if not results:
            return "No relevant information found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"Source {i}: {result['title']}\n{result['content']}\nURL: {result['url']}"
            )
        
        return "\n\n".join(context_parts)
    
    def get_stats(self) -> Dict:
        """Return cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "rate_limit_calls_remaining": self.rate_limiter.max_calls - len(self.rate_limiter.local_calls) # Local fallback context
        }


# Global instance and lock
import threading
_tavily_client_lock = threading.Lock()
_tavily_client: Optional[CachedTavilyRAG] = None


def get_tavily_client() -> CachedTavilyRAG:
    """Get or create Tavily client instance."""
    global _tavily_client
    if _tavily_client is None:
        with _tavily_client_lock:
            if _tavily_client is None:
                _tavily_client = CachedTavilyRAG()
    return _tavily_client

