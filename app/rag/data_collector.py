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
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "context": context,
            "response": response,
            "intent": intent,
            "feedback": feedback,
            "confidence": confidence,
            "source": source,
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

    async def cache_response(self, query: str, response: str, ttl: int = 300):
        """Cache high-confidence response in Redis."""
        if self.redis_client:
            try:
                import hashlib
                query_hash = hashlib.md5(query.strip().lower().encode()).hexdigest()
                key = f"cache:response:{query_hash}"
                await self.redis_client.set(key, response, ex=ttl)
            except Exception as e:
                print(f"Redis cache set failed: {e}")

    async def get_cached_response(self, query: str) -> Optional[str]:
        """Retrieve response from Redis cache."""
        if self.redis_client:
            try:
                import hashlib
                from app.metrics.prometheus import redis_cache_hit_total, redis_cache_miss_total
                
                query_hash = hashlib.md5(query.strip().lower().encode()).hexdigest()
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
