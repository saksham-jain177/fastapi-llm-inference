"""
Adapter manager for loading and managing multiple LoRA adapters.
Supports dynamic adapter loading based on domain classification.
"""

from typing import Dict, Optional, List
from pathlib import Path
import os


class AdapterManager:
    """
    Manages multiple LoRA adapters for different domains.
    Loads adapters on-demand and caches them.
    """
    
    # Map domains to adapter paths
    ADAPTER_REGISTRY = {
        "code": "training/training/lora-adapter/checkpoint-63",  # Existing code adapter
        # Add more adapters here as they're trained
        # "medical": "training/medical-adapter",
        # "legal": "training/legal-adapter",
    }
    
    def __init__(self):
        """Initialize adapter manager."""
        self.loaded_adapters = {}
        self.model_cache = {}
        
        # Get project root
        self.project_root = Path(__file__).parent.parent.parent
        
        print(f"Adapter manager initialized with {len(self.ADAPTER_REGISTRY)} registered adapters")
    
    def get_adapter_path(self, domain: str) -> Optional[str]:
        """
        Get absolute path to adapter for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Absolute path to adapter directory, or None if not available
        """
        if domain not in self.ADAPTER_REGISTRY:
            return None
        
        relative_path = self.ADAPTER_REGISTRY[domain]
        absolute_path = self.project_root / relative_path
        
        if absolute_path.exists():
            return str(absolute_path)
        
        return None
    
    def has_adapter(self, domain: str) -> bool:
        """
        Check if adapter exists for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            True if adapter is available
        """
        return self.get_adapter_path(domain) is not None
    
    def get_available_domains(self) -> List[str]:
        """
        Get list of domains with trained adapters.
        
        Returns:
            List of domain names
        """
        available = []
        for domain in self.ADAPTER_REGISTRY:
            if self.has_adapter(domain):
                available.append(domain)
        return available
    
    def load_adapter_model(self, domain: str):
        """
        Load LoRA model for a specific domain.
        
        Args:
            domain: Domain name
            
        Returns:
            (model, tokenizer) tuple
        """
        # Check cache
        if domain in self.model_cache:
            return self.model_cache[domain]
        
        adapter_path = self.get_adapter_path(domain)
        if not adapter_path:
            raise ValueError(f"No adapter available for domain: {domain}")
        
        # Import here to avoid circular dependency
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        import torch
        
        MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
        
        print(f"Loading {domain} adapter from {adapter_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        
        # 4-bit config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        
        # Cache
        self.model_cache[domain] = (model, tokenizer)
        
        print(f"Loaded {domain} adapter successfully")
        return model, tokenizer
    
    def generate_with_adapter(self, domain: str, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Generate response using domain-specific adapter.
        
        Args:
            domain: Domain name
            prompt: User prompt
            max_new_tokens: Max tokens to generate
            
        Returns:
            Generated response
        """
        import torch
        
        model, tokenizer = self.load_adapter_model(domain)
        
        # Format as instruction
        instruction_prompt = f"Instruction: {prompt}\nResponse:"
        
        inputs = tokenizer(instruction_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.5,
                do_sample=True,
                top_p=0.85,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(instruction_prompt):].strip()
        
        return response


# Global instance
_adapter_manager = None


def get_adapter_manager() -> AdapterManager:
    """Get or create adapter manager instance."""
    global _adapter_manager
    if _adapter_manager is None:
        _adapter_manager = AdapterManager()
    return _adapter_manager
