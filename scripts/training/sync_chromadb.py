"""
Synchronize feedback data from MongoDB to ChromaDB.
Used to keep the vector store aligned with human preference data.
"""
import sys
from pathlib import Path
import os

# Execution Guardrail
if os.getenv("ALLOW_EXPLICIT_EXECUTION") != "true":
    raise RuntimeError(
        "Script execution not allowed. Set ALLOW_EXPLICIT_EXECUTION=true to proceed. "
        "This tool synchronizes human preference data into the vector store."
    )

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.feedback_retriever import get_feedback_retriever

if __name__ == "__main__":
    print("Syncing MongoDB feedback to ChromaDB...")
    retriever = get_feedback_retriever()
    retriever.sync_from_mongodb()
    print("✅ Sync complete!")
