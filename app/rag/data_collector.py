import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib
from app.observability.logger import get_logger

logger = get_logger()

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
                logger.info("DataCollector: Connected to Redis")
            except Exception as e:
                logger.error("DataCollector: Failed to connect to Redis", error=str(e))

        # Initialize ChromaDB Semantic Cache
        self.chroma_client = None
        self.semantic_cache = None
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Use persistent Chroma instance in data/chroma
            chroma_path = project_root / "data" / "chroma"
            chroma_path.mkdir(parents=True, exist_ok=True)
            
            # Check environment to see if we should use mocked/deterministic embedding
            use_deterministic = os.getenv("USE_DETERMINISTIC_INFERENCE", "false").lower() == "true"
            
            self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
            
            # Create or get the cache collection. 
            # Chroma automatically uses exactly the same all-MiniLM-L6-v2 embedding model 
            # by default under the hood unless we override it.
            self.semantic_cache = self.chroma_client.get_or_create_collection(
                name="rag_semantic_cache",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("DataCollector: Initialized ChromaDB semantic cache", path=str(chroma_path))
            
        except ImportError:
            logger.warning("DataCollector: chromadb not installed, semantic caching disabled")
        except Exception as e:
            logger.error("DataCollector: Failed to initialized ChromaDB semantic cache", error=str(e))

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
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
                return
            except Exception as e:
                import traceback
                logger.error("Mongo Log Error", error=str(e))
                # Guardrail: Fallback to local file logging is gated by ALLOW_LOCAL_FALLBACK to ensure Mongo-first policy by default.
                if os.getenv("ALLOW_LOCAL_FALLBACK", "false").lower() != "true":
                     logger.warning("Suggest enabling ALLOW_LOCAL_FALLBACK=true if Mongo is unstable.")
                     return
        else:
             if os.getenv("ALLOW_LOCAL_FALLBACK", "false").lower() != "true":
                 return
        
        # Fallback to File
        try:
            import aiofiles
            async with aiofiles.open(self.log_file, "a", encoding="utf-8") as f:
                await f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.error("Error logging interaction", error=str(e))

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
                logger.error("Error fetching from Mongo", error=str(e))
        
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
                logger.error("Error fetching from Mongo", error=str(e))
        
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
        """Cache high-confidence response in Semantic Cache (ChromaDB) and exact-match Redis cache."""
        # 1. Exact Match via Redis (Legacy/Fast fallback)
        if self.redis_client:
            try:
                query_hash = hashlib.sha256(query.strip().lower().encode()).hexdigest()
                key = f"cache:response:{query_hash}"
                await self.redis_client.set(key, response, ex=ttl)
            except Exception as e:
                logger.error("Redis cache set failed", error=str(e))
                
        # 2. Semantic Match via ChromaDB
        if self.semantic_cache:
            try:
                doc_id = hashlib.sha256(query.strip().lower().encode()).hexdigest()
                
                # We offload Chroma calls to a thread since it does disk I/O / sync ops
                import asyncio
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.semantic_cache.upsert(
                        ids=[doc_id],
                        documents=[query],
                        metadatas=[{"response": response, "created_at": datetime.now(timezone.utc).isoformat()}]
                    )
                )
            except Exception as e:
                logger.error("ChromaDB semantic cache set failed", error=str(e))

    async def get_cached_response(self, query: str) -> Optional[str]:
        """Retrieve response from Semantic Cache, falling back to Redis exact-match."""
        
        # 1. Check Semantic Cache (ChromaDB)
        if self.semantic_cache:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                
                results = await loop.run_in_executor(
                    None,
                    lambda: self.semantic_cache.query(
                        query_texts=[query],
                        n_results=1
                    )
                )
                
                # Check distances (cosine distance - closer to 0 is better)
                if results and results['distances'] and len(results['distances'][0]) > 0:
                    distance = results['distances'][0][0]
                    # Score < 0.15 is highly semantically similar
                    if distance < 0.15:
                        cached_response = results['metadatas'][0][0]['response']
                        from app.metrics.prometheus import semantic_cache_hit_total
                        semantic_cache_hit_total.inc()
                        logger.info("Semantic Cache Hit", query=query, distance=distance)
                        return cached_response
                        
            except Exception as e:
                logger.error("ChromaDB semantic cache get failed", error=str(e))
                
        # 2. Check Exact Match Cache (Redis)
        if self.redis_client:
            try:
                from app.metrics.prometheus import redis_cache_hit_total, redis_cache_miss_total
                
                query_hash = hashlib.sha256(query.strip().lower().encode()).hexdigest()
                key = f"cache:response:{query_hash}"
                cached = await self.redis_client.get(key)
                
                if cached:
                    redis_cache_hit_total.inc()
                    return cached
                else:
                    redis_cache_miss_total.inc()
                    
            except Exception as e:
                logger.error("Redis cache get failed", error=str(e))
                
        return None

# Global instance and lock
import threading
_collector_lock = threading.Lock()
_collector = None

def get_data_collector() -> DataCollector:
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = DataCollector()
    return _collector
