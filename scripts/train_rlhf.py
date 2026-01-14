import json
import os
from pathlib import Path
import pandas as pd
from datasets import Dataset

# Configuration
DATA_DIR = Path("training/collected_data")
LOG_FILE = DATA_DIR / "rag_interactions.jsonl"
OUTPUT_FILE = DATA_DIR / "processed_rlhf_dataset.json"

def load_data():
    """Load raw interaction logs."""
    if not LOG_FILE.exists():
        print(f"No data found at {LOG_FILE}")
        return []
    
    data = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                # Only keep entries with explicit feedback
                if entry.get("feedback"):
                    data.append(entry)
            except json.JSONDecodeError:
                continue
    return data

def prepare_kto_dataset(raw_data):
    """
    Prepare data for KTO with corrected abstention rewards.
    
    Prevents leakage by:
    - Only rewarding ACTUAL abstentions (not wrong answers marked "should_refuse")
    - Penalizing answers that should have refused
    
    Format required:
    - prompt: str
    - completion: str
    - label: bool (True=Good, False=Bad)
    """
    from app.constants import CANONICAL_REFUSAL
    
    processed = []
    for entry in raw_data:
        feedback_label = entry.get("feedback_label")
        response = entry.get("response", "")
        
        if not feedback_label:
            # Old format fallback (binary feedback)
            rating = int(entry.get("feedback", 0))
            if rating == 0:
                continue
            processed.append({
                "prompt": entry["query"],
                "completion": response,
                "label": rating > 0,
                "context": entry.get("context", "")
            })
            continue
        
        # Check if model actually abstained
        model_abstained = (response.strip() == CANONICAL_REFUSAL)
        
        # Label: correct
        if feedback_label == "correct":
            processed.append({
                "prompt": entry["query"],
                "completion": response,
                "label": True,  # Positive
                "context": entry.get("context", "")
            })
        
        # Label: should_have_refused
        elif feedback_label == "should_have_refused":
            if model_abstained:
                # Model correctly refused - REWARD
                processed.append({
                    "prompt": entry["query"],
                    "completion": CANONICAL_REFUSAL,
                    "label": True,  # Positive - teaches refusal
                    "context": entry.get("context", "")
                })
            else:
                # Model answered when it should have refused - PENALIZE
                processed.append({
                    "prompt": entry["query"],
                    "completion": response,
                    "label": False,  # Negative - punish overconfidence
                    "context": entry.get("context", "")
                })
        
        # Label: incorrect
        elif feedback_label == "incorrect":
            processed.append({
                "prompt": entry["query"],
                "completion": response,
                "label": False,  # Negative
                "context": entry.get("context", "")
            })
    
    return processed

def main():
    print(f"Loading data from {LOG_FILE}...")
    raw_data = load_data()
    print(f"Found {len(raw_data)} interactions with feedback.")
    
    if not raw_data:
        print("No training data available. Go use the chat interface and vote!")
        return

    # Process for KTO
    kto_data = prepare_kto_dataset(raw_data)
    
    # Convert to HuggingFace Dataset (standard for training)
    ds = Dataset.from_list(kto_data)
    
    print("\n--- Sample KTO Entry ---")
    print(json.dumps(kto_data[0], indent=2))
    
    # Save processed dataset
    ds.to_json(OUTPUT_FILE)
    print(f"\nSaved processed dataset to {OUTPUT_FILE}")
    print(f"Ready for Human-Centered Loss (KTO) training!")

    # Example of how to run training (Mock Code)
    print("\n--- To Run Training (Example) ---")
    print("""
    from trl import KTOTrainer, KTOConfig
    
    config = KTOConfig(
        beta=0.1,
        learning_rate=5e-7,
        output_dir="training/models/rlhf_v1"
    )
    
    trainer = KTOTrainer(
        model=model,
        ref_model=ref_model,
        train_dataset=ds,
        args=config
    )
    trainer.train()
    """)

if __name__ == "__main__":
    main()
