"""
Semantic Feedback Retriever using ChromaDB.

Stores past interactions with embeddings for similarity search.
When user asks a question, retrieves similar past Q&A pairs with positive feedback.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Optional

class FeedbackRetriever:
    def __init__(self):
        """Initialize ChromaDB with persistent storage."""
        import os
        self.use_deterministic = os.getenv("USE_DETERMINISTIC_INFERENCE", "false").lower() == "true"
        
        if not self.use_deterministic:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            self.encoder = None
        
        # Persistent storage in project directory
        db_path = os.path.join(os.path.dirname(__file__), '../../data/chroma')
        os.makedirs(db_path, exist_ok=True)
        
        # Use simple settings for persistent client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="feedback_memory",
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
    
    def add_interaction(self, query: str, response: str, feedback: str, interaction_id: str):
        """
        Store interaction with embedding.
        
        Args:
            query: User's question
            response: Model's answer
            feedback: "1" (positive) or "-1" (negative)
            interaction_id: Unique ID from MongoDB
        """
        import time
        # Store all feedback, positive or negative
        
        # Generate embedding
        embedding = self.encoder.encode(query).tolist()
        
        # Store in ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[response],
            metadatas=[{
                "query": query,
                "feedback": feedback,
                "timestamp": time.time()
            }],
            ids=[interaction_id]
        )
    
    def search_similar(self, query: str, top_k: int = 3, min_similarity: float = 0.7) -> List[Dict]:
        """
        Search for similar past queries with composite confidence scoring.
        
        Args:
            query: Current user query
            top_k: Number of results to return
            min_similarity: Minimum cosine similarity threshold (0-1)
        
        Returns:
            List of {query, response, similarity, confidence, ...} dicts
        """
        if self.use_deterministic:
            return []

        # Generate embedding for query
        query_embedding = self.encoder.encode(query).tolist()
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # Fetch more to allow for suppression filtering
            include=['metadatas', 'documents', 'distances']
        )
        
        # Format results
        matches = []
        import time
        import math
        
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                # 1. Base Similarity
                distance = results['distances'][0][i]
                similarity = 1 - distance  # Cosine distance -> similarity
                if similarity < min_similarity:
                    continue
                    
                meta = results['metadatas'][0][i]
                response_text = results['documents'][0][i]
                
                # 2. Feedback Suppression check (Simple entry-level check)
                # If THIS specific interaction was negative, we suppress it heavily
                feedback_val = meta.get("feedback", "1")
                if feedback_val == "-1":
                    # Suppressed: Net negative feedback for this entry
                    print(f"  [Memory] Suppressing negative entry (id={results['ids'][0][i]})")
                    continue

                # 3. Recency Weight (Exponential Decay)
                # Half-life of ~7 days? Let's say 30 days.
                # lambda = ln(2) / half_life
                # weight = e^(-lambda * elapsed_days)
                timestamp = meta.get("timestamp", 0)
                if timestamp == 0:
                     # Legacy entries without timestamp get neutral weight
                     recency_weight = 0.8 
                else:
                    elapsed_seconds = time.time() - timestamp
                    elapsed_days = elapsed_seconds / 86400
                    lambda_decay = 0.693 / 30  # 30 day half-life
                    recency_weight = math.exp(-lambda_decay * elapsed_days)
                
                # 4. Composite Confidence Score
                # effective_confidence = similarity * feedback_weight * recency_weight
                # feedback_weight is 1.0 since we filtered negatives
                effective_confidence = similarity * 1.0 * recency_weight
                
                matches.append({
                    "query": meta.get("query", ""),
                    "response": response_text,
                    "similarity": round(similarity, 3),
                    "confidence": round(effective_confidence, 3),
                    "recency": round(recency_weight, 3),
                    "timestamp": timestamp
                })
        
        # Sort by effective confidence
        matches.sort(key=lambda x: x["confidence"], reverse=True)
        return matches[:top_k]
    
    def sync_from_mongodb(self):
        """
        One-time sync: Load all positive feedback from MongoDB into ChromaDB.
        Used to initialize semantic memory from seed data or collective logs.
        """
        from app.rag.data_collector import get_data_collector
        import asyncio
        
        async def _sync():
            collector = get_data_collector()
            
            # Get all interactions from MongoDB
            if collector.mongo_collection is not None:
                cursor = collector.mongo_collection.find({"feedback": "1"})
                count = 0
                
                async for doc in cursor:
                    self.add_interaction(
                        query=doc['query'],
                        response=doc['response'],
                        feedback=doc['feedback'],
                        interaction_id=str(doc['_id'])
                    )
                    count += 1
                
                print(f"✅ Synced {count} positive interactions to ChromaDB")
        
        asyncio.run(_sync())


# Singleton instance
_retriever_instance = None

def get_feedback_retriever() -> FeedbackRetriever:
    """Get singleton instance of FeedbackRetriever."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = FeedbackRetriever()
    return _retriever_instance
