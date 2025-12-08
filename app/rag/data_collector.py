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
        # Resolve to data_dir relative to project root (assuming app/rag/data_collector.py)
        # We want f:\Projects\fastapi-llm-inference\training\collected_data
        project_root = Path(__file__).resolve().parents[2] 
        self.log_dir = project_root / "training" / "collected_data"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "rag_interactions.jsonl"
        
    def log_interaction(self, 
                       query: str, 
                       context: str, 
                       response: str, 
                       intent: str,
                       feedback: Optional[str] = None):
        """
        Log a single interaction to the JSONL file.
        
        Args:
            query: User's question
            context: Retrieved context used to answer
            response: Model's generated answer
            intent: Classification intent
            feedback: Optional user feedback (positive/negative)
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "context": context,
            "response": response,
            "intent": intent,
            "feedback": feedback,
            # Format for Qwen fine-tuning (alpaca style or similar)
            "training_sample": {
                "instruction": query,
                "input": context,
                "output": response
            }
        }
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error logging interaction: {e}")

    def get_stats(self) -> Dict[str, int]:
        """Return count of collected samples."""
        if not self.log_file.exists():
            return {"count": 0}
            
        with open(self.log_file, "r", encoding="utf-8") as f:
            count = sum(1 for _ in f)
            return {"count": count}

# Global instance
_collector = None

def get_data_collector() -> DataCollector:
    global _collector
    if _collector is None:
        _collector = DataCollector()
    return _collector
