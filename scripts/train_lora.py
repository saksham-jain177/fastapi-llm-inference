"""
LoRA fine-tuning script for Qwen2.5-0.5B-Instruct.
Uses PEFT library with 4-bit quantization for efficient training.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "training/lora-adapter"
DATASET_PATH = "training/data/lora_dataset.json"


def format_instruction(example):
    """Format example as instruction prompt."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if input_text:
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
    else:
        prompt = f"Instruction: {instruction}\nResponse:"
    
    return {"text": f"{prompt} {output}"}


def load_and_prepare_data():
    """Load and tokenize dataset."""
    print(f"Loading dataset from {DATASET_PATH}")
    
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to HF dataset
    from datasets import Dataset
    dataset = Dataset.from_list(data)
    
    # Format instructions
    dataset = dataset.map(format_instruction)
    
    print(f"Loaded {len(dataset)} examples")
    return dataset


def train_lora():
    """Train LoRA adapter."""
    print("Initializing LoRA training...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # Alpha scaling
        target_modules=["q_proj", "v_proj"],  # Which layers to apply LoRA
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    dataset = load_and_prepare_data()
    
    # Tokenize
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        # Add labels for causal LM (same as input_ids)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    # Training arguments (optimized for speed)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,  # Reduced from 3 for faster training
        per_device_train_batch_size=8,  # Increased batch size
        gradient_accumulation_steps=2,  # Reduced accumulation
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=1,  # Keep only final checkpoint
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save adapter
    print(f"Saving LoRA adapter to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training complete!")


if __name__ == "__main__":
    train_lora()
