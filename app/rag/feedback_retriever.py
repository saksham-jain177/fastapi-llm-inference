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
        # Reuse same embeddings as semantic router for consistency
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Persistent storage in project directory
        db_path = os.path.join(os.path.dirname(__file__), '../../data/chroma')
        os.makedirs(db_path, exist_ok=True)
        
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
        # Only store positive feedback
        if feedback != "1":
            return
        
        # Generate embedding
        embedding = self.encoder.encode(query).tolist()
        
        # Store in ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[response],
            metadatas=[{
                "query": query,
                "feedback": feedback
            }],
            ids=[interaction_id]
        )
    
    def search_similar(self, query: str, top_k: int = 3, min_similarity: float = 0.7) -> List[Dict]:
        """
        Search for similar past queries with positive feedback.
        
        Args:
            query: Current user query
            top_k: Number of results to return
            min_similarity: Minimum cosine similarity threshold (0-1)
        
        Returns:
            List of {query, response, similarity} dicts
        """
        # Generate embedding for query
        query_embedding = self.encoder.encode(query).tolist()
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['metadatas', 'documents', 'distances']
        )
        
        # Format results
        matches = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                # ChromaDB returns distance (lower = more similar)
                # Convert to similarity score
                distance = results['distances'][0][i]
                similarity = 1 - distance  # Cosine distance -> similarity
                
                if similarity >= min_similarity:
                    matches.append({
                        "query": results['metadatas'][0][i]["query"],
                        "response": results['documents'][0][i],
                        "similarity": round(similarity, 3)
                    })
        
        return matches
    
    def sync_from_mongodb(self):
        """
        One-time sync: Load all positive feedback from MongoDB into ChromaDB.
        Call this after populating MongoDB with synthetic data.
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
