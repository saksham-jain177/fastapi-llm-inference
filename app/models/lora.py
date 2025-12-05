"""
LoRA inference loader - loads base model with trained LoRA adapter.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Optional
from pathlib import Path
import os

# Global instances
_lora_model = None
_lora_tokenizer = None

# Get project root (app/models/lora.py -> app/models -> app -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_ADAPTER_PATH = str(PROJECT_ROOT / "training" / "training" / "lora-adapter" / "checkpoint-63")


def get_bnb_config() -> BitsAndBytesConfig:
    """4-bit quantization config."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_lora_model():
    """Load base model with LoRA adapter."""
    global _lora_model, _lora_tokenizer
    
    if _lora_model is not None and _lora_tokenizer is not None:
        return _lora_model, _lora_tokenizer
    
    print(f"Loading LoRA model from {LORA_ADAPTER_PATH}")
    
    # Load tokenizer
    _lora_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _lora_tokenizer.pad_token = _lora_tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load LoRA adapter
    _lora_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    _lora_model.eval()
    
    print("LoRA model loaded successfully!")
    return _lora_model, _lora_tokenizer


def generate_lora_response(prompt: str, max_new_tokens: int = 256, temperature: float = 0.5) -> str:
    """Generate response using LoRA-tuned model."""
    model, tokenizer = load_lora_model()
    
    # Format as instruction
    instruction_prompt = f"Instruction: {prompt}\nResponse:"
    
    inputs = tokenizer(instruction_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.85,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response after prompt
    response = full_response[len(instruction_prompt):].strip()
    
    return response


def get_lora_model_info() -> dict:
    """Get LoRA model info."""
    adapter_exists = os.path.exists(LORA_ADAPTER_PATH)
    
    if not adapter_exists:
        return {"loaded": False, "adapter_path": LORA_ADAPTER_PATH, "exists": False}
    
    if _lora_model is None:
        return {"loaded": False, "adapter_path": LORA_ADAPTER_PATH, "exists": True}
    
    return {
        "loaded": True,
        "model_name": MODEL_NAME,
        "adapter_path": LORA_ADAPTER_PATH,
        "device": str(_lora_model.device),
        "quantization": "4-bit (bitsandbytes nf4)",
    }
