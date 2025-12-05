"""
Download and prepare dataset for LoRA fine-tuning.
Uses flytech/python-codes-25k - high-quality Python instruction-output pairs.
"""

from datasets import load_dataset
import json
from pathlib import Path


def prepare_dataset(output_path: str = "training/data/lora_dataset.json", max_samples: int = 1000):
    """
    Download and format dataset for LoRA training.
    
    Args:
        output_path: Path to save formatted dataset
        max_samples: Maximum number of samples to use
    """
    print(f"Downloading dataset (max {max_samples} samples)...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("flytech/python-codes-25k", split=f"train[:{max_samples}]")
    
    formatted_data = []
    
    for item in dataset:
        # Format as instruction-response pairs
        formatted_data.append({
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", ""),
        })
    
    # Save to JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"Total samples: {len(formatted_data)}")
    
    # Print sample
    if formatted_data:
        print("\nSample entry:")
        print(json.dumps(formatted_data[0], indent=2))


if __name__ == "__main__":
    prepare_dataset()
