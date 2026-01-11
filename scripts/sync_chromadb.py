"""
One-time script to sync MongoDB feedback into ChromaDB.
Run this after generating synthetic data.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.feedback_retriever import get_feedback_retriever

if __name__ == "__main__":
    print("Syncing MongoDB feedback to ChromaDB...")
    retriever = get_feedback_retriever()
    retriever.sync_from_mongodb()
    print("✅ Sync complete!")
