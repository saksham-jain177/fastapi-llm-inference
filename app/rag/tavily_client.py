"""
Tavily RAG client with caching, rate limiting, and error handling.
"""

import os
import time
import hashlib
from tavily import TavilyClient
from typing import List, Dict, Optional
from functools import lru_cache
from datetime import datetime, timedelta


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int = 10, period_seconds: int = 60):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.calls = []
    
    def allow_request(self) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        # Remove calls outside the window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.period_seconds]
        
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False
    
    def wait_time(self) -> float:
        """Return seconds to wait before next allowed request."""
        if not self.calls:
            return 0.0
        oldest_call = min(self.calls)
        return max(0.0, self.period_seconds - (time.time() - oldest_call))


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
        return hashlib.md5(content.encode()).hexdigest()
    
    def _add_to_cache(self, key: str, value: List[Dict]):
        """Add to cache with LRU eviction."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO, not true LRU but sufficient)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
    
    def search(self, query: str, max_results: int = 3, max_retries: int = 3) -> List[Dict]:
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
        
        # Rate limiting
        if not self.rate_limiter.allow_request():
            wait_time = self.rate_limiter.wait_time()
            print(f"Rate limit reached. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                response = self.client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=max_results
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
                    print(f"Tavily API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    print(f"Tavily API error after {max_retries} attempts: {e}")
                    return []
    
    def get_context(self, query: str, max_results: int = 3) -> str:
        """Get formatted context string from search results."""
        results = self.search(query, max_results)
        
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
            "rate_limit_calls_remaining": self.rate_limiter.max_calls - len(self.rate_limiter.calls)
        }


# Global instance
_tavily_client: Optional[CachedTavilyRAG] = None


def get_tavily_client() -> CachedTavilyRAG:
    """Get or create Tavily client instance."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = CachedTavilyRAG()
    return _tavily_client

