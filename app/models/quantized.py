"""
Quantized model loader using bitsandbytes 4-bit quantization.
Uses Qwen2.5-0.5B-Instruct for efficient inference on consumer GPUs.
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from typing import Optional
from threading import Thread
from tqdm import tqdm

# Global model and tokenizer instances (loaded once)
_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def get_bnb_config() -> BitsAndBytesConfig:
    """
    Configure 4-bit quantization with bitsandbytes.
    This reduces VRAM usage from ~2GB to ~500MB.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
        bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
    )


def load_model():
    """
    Load the quantized model and tokenizer.
    This is called once at startup and cached globally.
    """
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    print(f"Loading quantized model: {MODEL_NAME}")
    
    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model with 4-bit quantization
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=get_bnb_config(),
        device_map="auto",  # Automatically distribute across available devices
        trust_remote_code=True,
    )
    
    print("Model loaded successfully!")
    print(f"Device: {_model.device}")
    
    return _model, _tokenizer


def generate_response(prompt: str, max_new_tokens: int = 256, temperature: float = 0.5) -> str:
    """
    Generate a response using the quantized model.
    
    Args:
        prompt: User's input prompt
        max_new_tokens: Maximum tokens to generate (default 256 for fuller responses)
        temperature: Sampling temperature (lower = more focused, less hallucination)
    
    Returns:
        Generated text response
    """
    model, tokenizer = load_model()
    
    # Format prompt for Qwen chat template
    messages = [
        {"role": "system", "content": "You are a helpful and accurate AI assistant. Provide concise, factual responses."},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate (progress bar suppressed for cleaner logs)
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
    
    # Decode and extract response
    full_response_with_tokens = tokenizer.decode(outputs[0], skip_special_tokens=False)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant's response
    assistant_markers = [
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "assistant\n",
    ]
    
    response = None
    for marker in assistant_markers:
        if marker in full_response_with_tokens:
            parts = full_response_with_tokens.split(marker, 1)
            if len(parts) > 1:
                response = parts[1].split("<|im_end|>")[0].strip()
                break
    
    # Fallback extraction
    if not response:
        prompt_idx = full_response.lower().find(prompt.lower())
        if prompt_idx != -1:
            response = full_response[prompt_idx + len(prompt):].strip()
        else:
            response = full_response.strip()
    
    return response


def generate_stream(prompt: str, max_new_tokens: int = 512, temperature: float = 0.1):
    """
    Generator function that yields tokens as they are generated.
    """
    model, tokenizer = load_model()
    
    messages = [
        {"role": "system", "content": "You are a precise and helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs, 
        streamer=streamer, 
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.1  # Reduce repetitive verbosity
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    for new_text in streamer:
        yield new_text


def get_model_info() -> dict:
    """Get information about the loaded model."""
    if _model is None:
        return {"loaded": False}
    
    return {
        "loaded": True,
        "model_name": MODEL_NAME,
        "device": str(_model.device),
        "quantization": "4-bit (bitsandbytes nf4)",
        "dtype": str(_model.dtype),
    }

