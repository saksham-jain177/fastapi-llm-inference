import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class DataCollector:
    """
    Collects successful RAG interactions to build a fine-tuning dataset.
    Implements the "A+B=AB" continuous learning loop.
    """
    
    def __init__(self, log_dir: str = "training/collected_data"):
        # Resolve to data_dir relative to project root
        project_root = Path(__file__).resolve().parents[2] 
        self.log_dir = project_root / "training" / "collected_data"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "rag_interactions.jsonl"
        
        # Initialize Redis/Mongo optional connections
        self.mongo_collection = None
        mongo_url = os.getenv("MONGO_URL")
        if mongo_url:
            try:
                from motor.motor_asyncio import AsyncIOMotorClient
                client = AsyncIOMotorClient(mongo_url)
                db = client.llm_inference
                self.mongo_collection = db.interactions
                print("DataCollector: Connected to MongoDB")
            except Exception as e:
                print(f"DataCollector: Failed to connect to Mongo, falling back to file. {e}")

        # Initialize Redis
        self.redis_client = None
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                import redis.asyncio as redis
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                print("DataCollector: Connected to Redis")
            except Exception as e:
                print(f"DataCollector: Failed to connect to Redis. {e}")

        # ---- Semantic cache config (embedding cosine-similarity lookup) ----
        # SEMANTIC_CACHE_THRESHOLD: min cosine similarity for a cache hit.
        # SEMANTIC_CACHE_MAX_ENTRIES: max entries scanned per lookup (bounded scan).
        self.semantic_cache_enabled = (
            os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
        )
        self.semantic_threshold = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
        self.semantic_max_entries = int(os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "500"))
        self._embedding_model = None

    async def log_interaction(self, 
                       query: str, 
                       context: str, 
                       response: str, 
                       intent: str,
                       feedback: Optional[str] = None,
                       confidence: float = 0.0,
                       source: str = "model"):
        """
        Log a single interaction to MongoDB (primary) or JSONL (backup).
        """
        # Calculate metadata
        char_count = len(response)
        word_count = len(response.split())
        
        # Truncation detected via same logic as orchestrator
        is_truncated = False
        if response:
            clean_resp = response.strip()
            valid_endings = ('.', '!', '?', '"', "'", '`', '}')
            trailing_indicators = ('and', 'or', 'but', 'the', 'a', 'an', 'with', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'which', 'that')
            if not clean_resp.endswith(valid_endings) or clean_resp.lower().endswith(trailing_indicators):
                is_truncated = True

        conf_bucket = "low"
        if confidence >= 0.85: conf_bucket = "high"
        elif confidence >= 0.6: conf_bucket = "medium"

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "context": context,
            "response": response,
            "intent": intent,
            "feedback": feedback,
            "confidence": confidence,
            "confidence_bucket": conf_bucket,
            "source": source,
            "metadata": {
                "char_length": char_count,
                "word_length": word_count,
                "is_truncated": is_truncated
            },
            # Format for Qwen fine-tuning
            "training_sample": {
                "instruction": query,
                "input": context,
                "output": response
            }
        }
        
        # Strategy: Try Mongo First
        if self.mongo_collection is not None:
            try:
                await self.mongo_collection.insert_one(entry)
                from app.metrics.prometheus import mongodb_write_total
                mongodb_write_total.inc()
                # print(f"✅ MongoDB: Saved feedback to MongoDB")
                return
            except Exception as e:
                import traceback
                print(f"❌ Mongo Log Error: {e}")
                # Guardrail: Fallback to local file logging is gated by ALLOW_LOCAL_FALLBACK to ensure Mongo-first policy by default.
                if os.getenv("ALLOW_LOCAL_FALLBACK", "false").lower() != "true":
                     print("Suggest enabling ALLOW_LOCAL_FALLBACK=true if Mongo is unstable.")
                     return
        else:
             if os.getenv("ALLOW_LOCAL_FALLBACK", "false").lower() != "true":
                 return
        
        # Fallback to File
        try:
            # Note: File writing is sync/blocking, but acceptable for fallback
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            print(f"Error logging interaction: {e}")

    async def get_stats(self) -> Dict[str, int]:
        """Return count of collected samples."""
        count = 0
        if self.mongo_collection is not None:
            try:
                count = await self.mongo_collection.count_documents({})
                return {"count": count, "source": "mongo"}
            except:
                pass
                
        if self.log_file.exists():
            with open(self.log_file, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f)
        
        return {"count": count, "source": "file"}

    async def get_recent(self, limit: int = 50) -> list:
        """Fetch recent interactions for logs viewer."""
        logs = []
        
        # Try Mongo first
        if self.mongo_collection is not None:
            try:
                cursor = self.mongo_collection.find().sort("timestamp", -1).limit(limit)
                async for doc in cursor:
                    doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
                    logs.append(doc)
                return logs
            except Exception as e:
                print(f"Error fetching from Mongo: {e}")
        
        # Fallback to file only if allowed
        if os.getenv("ALLOW_LOCAL_FALLBACK", "false").lower() == "true":
            if self.log_file.exists():
                with open(self.log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in reversed(lines[-limit:]):
                        try:
                            logs.append(json.loads(line))
                        except:
                            pass
        
        return logs

    async def get_all_interactions(self) -> list:
        """
        Fetch ALL interactions for threshold calibration.
        Used by confidence threshold auto-calibration.
        """
        interactions = []
        
        # Try Mongo first
        if self.mongo_collection is not None:
            try:
                cursor = self.mongo_collection.find({})
                async for doc in cursor:
                    interactions.append(doc)
                return interactions
            except Exception as e:
                print(f"Error fetching from Mongo: {e}")
        
        # Fallback to file only if allowed
        if os.getenv("ALLOW_LOCAL_FALLBACK", "false").lower() == "true":
            if self.log_file.exists():
                with open(self.log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            interactions.append(json.loads(line))
                        except:
                            continue
        
        return interactions

    def _get_embedding_model(self):
        """Lazily load the shared sentence-transformer model.

        Returns None in deterministic mode or when the model is unavailable,
        so callers can fall back to exact-match caching.
        """
        if not self.semantic_cache_enabled:
            return None
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
            except Exception as e:
                print(f"Semantic cache: embedding model unavailable, "
                      f"falling back to exact match. {e}")
                self._embedding_model = False  # sentinel: failed load
        return self._embedding_model or None

    @staticmethod
    def _normalize_query(query: str) -> str:
        return query.strip().lower()

    async def _semantic_lookup(self, query: str):
        """
        Cosine-similarity lookup over cached query embeddings.

        Cache layout in Redis:
          cache:semantic:index        -> JSON list of {h: query_hash} entries
          cache:response:{query_hash} -> response text
          cache:embedding:{query_hash}-> JSON list of floats (query embedding)

        Returns (response, similarity) or (None, 0.0) on miss.
        """
        import json
        import hashlib
        import numpy as np

        model = self._get_embedding_model()
        if model is None or not self.redis_client:
            return None, 0.0

        try:
            index_raw = await self.redis_client.get("cache:semantic:index")
            if not index_raw:
                return None, 0.0
            index = json.loads(index_raw)[-self.semantic_max_entries:]

            q_emb = np.asarray(model.encode([self._normalize_query(query)])[0])

            best_hash, best_sim = None, -1.0
            for entry in index:
                h = entry.get("h")
                if not h:
                    continue
                raw = await self.redis_client.get(f"cache:embedding:{h}")
                if not raw:
                    continue  # embedding expired; skip stale index entry
                emb = np.asarray(json.loads(raw))
                denom = (np.linalg.norm(q_emb) * np.linalg.norm(emb))
                if denom == 0:
                    continue
                sim = float(np.dot(q_emb, emb) / denom)
                if sim > best_sim:
                    best_sim, best_hash = sim, h

            if best_hash is not None and best_sim >= self.semantic_threshold:
                response = await self.redis_client.get(f"cache:response:{best_hash}")
                if response is not None:
                    return response, best_sim
            return None, 0.0
        except Exception as e:
            print(f"Semantic cache lookup failed: {e}")
            return None, 0.0

    async def cache_response(self, query: str, response: str, ttl: int = 300):
        """Cache high-confidence response in Redis with a semantic index entry."""
        if self.redis_client:
            try:
                import hashlib
                import json

                norm = self._normalize_query(query)
                query_hash = hashlib.md5(norm.encode()).hexdigest()
                key = f"cache:response:{query_hash}"
                await self.redis_client.set(key, response, ex=ttl)

                model = self._get_embedding_model()
                if model is not None:
                    emb = [float(x) for x in model.encode([norm])[0]]
                    await self.redis_client.set(
                        f"cache:embedding:{query_hash}",
                        json.dumps(emb),
                        ex=ttl,
                    )
                    index_raw = await self.redis_client.get("cache:semantic:index")
                    try:
                        index = json.loads(index_raw) if index_raw else []
                    except Exception:
                        index = []
                    index = [e for e in index if e.get("h") != query_hash]
                    index.append({"h": query_hash})
                    await self.redis_client.set(
                        "cache:semantic:index",
                        json.dumps(index[-self.semantic_max_entries:]),
                        ex=max(ttl, 3600),
                    )
            except Exception as e:
                print(f"Redis cache set failed: {e}")

    async def get_cached_response(self, query: str) -> Optional[str]:
        """Retrieve response from Redis via semantic similarity (exact-match fallback)."""
        if self.redis_client:
            try:
                import hashlib
                from app.metrics.prometheus import redis_cache_hit_total, redis_cache_miss_total

                # 1. Semantic lookup first (higher hit rate).
                if self._get_embedding_model() is not None:
                    hit, _sim = await self._semantic_lookup(query)
                    if hit is not None:
                        redis_cache_hit_total.inc()
                        return hit

                # 2. Fallback / secondary: exact normalized match.
                query_hash = hashlib.md5(self._normalize_query(query).encode()).hexdigest()
                key = f"cache:response:{query_hash}"
                cached = await self.redis_client.get(key)

                if cached:
                    redis_cache_hit_total.inc()
                    return cached
                else:
                    redis_cache_miss_total.inc()
                    return None
            except Exception as e:
                print(f"Redis cache get failed: {e}")
                return None
        return None

# Global instance
_collector = None

def get_data_collector() -> DataCollector:
    global _collector
    if _collector is None:
        _collector = DataCollector()
    return _collector
