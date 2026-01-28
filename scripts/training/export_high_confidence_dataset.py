import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Defaults
DEFAULT_CONF_THRESHOLD = 0.85

async def export_curated_dataset(output_path: str = "training/curated_dataset.jsonl"):
    """
    Extracts high-confidence, non-search, non-refused interactions from MongoDB.
    Pure read-only operation.
    """
    # Evaluate at runtime to support dynamic testing
    allow_export = os.getenv("ALLOW_DATA_EXPORT", "false").lower() == "true"
    conf_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", str(DEFAULT_CONF_THRESHOLD)))

    if not allow_export:
        print("❌ ERROR: ALLOW_DATA_EXPORT=true is required to run this script.")
        return

    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file = output_file.with_suffix(".meta.json")

    # Access MongoDB via DataCollector
    from app.rag.data_collector import get_data_collector
    collector = get_data_collector()
    
    if collector.mongo_collection is None:
        print("❌ ERROR: MongoDB connection unavailable.")
        return

    print(f"🔍 Querying MongoDB for curated dataset (threshold >= {conf_threshold})...")
    
    # Filter Logic:
    # 1. confidence >= threshold
    # 2. refused == false
    # 3. source in [model, memory] (no external RAG noise)
    # 4. intent != external_search (focus on internal reasoning)
    query = {
        "confidence": {"$gte": conf_threshold},
        "refused": False,
        "source": {"$in": ["model", "memory"]},
        "intent": {"$ne": "external_search"}
    }
    
    cursor = collector.mongo_collection.find(query)
    
    # Deterministic Ordering: confidence (desc) -> timestamp (desc)
    # This prevents dataset drift and implements a quality-first curriculum implicitly
    cursor = cursor.sort([("confidence", -1), ("timestamp", -1)])
    
    exported_count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        async for doc in cursor:
            # Map to Qwen LoRA Schema
            sample = {
                "instruction": doc.get("query", ""),
                "input": doc.get("context", ""),
                "output": doc.get("response", "")
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            exported_count += 1

    # sidecar metadata
    metadata = {
        "exported_at": datetime.utcnow().isoformat(),
        "confidence_threshold": conf_threshold,
        "row_count": exported_count,
        "filters": {
            "refused": False,
            "sources": ["model", "memory"],
            "excluded_intents": ["external_search"]
        },
        "filename": output_file.name
    }
    
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Success! Exported {exported_count} samples to {output_file}")
    print(f"📄 Metadata saved to {meta_file}")

if __name__ == "__main__":
    asyncio.run(export_curated_dataset())
