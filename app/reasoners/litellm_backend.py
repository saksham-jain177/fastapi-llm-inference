"""
LiteLLM Reasoner backend.
Routes all LLM inference through LiteLLM's standard API, enabling high-concurrency external endpoints (vLLM) and instant fallback to cloud providers (OpenAI, Anthropics, etc).
"""

import os
import json
from typing import Dict, AsyncGenerator
from litellm import acompletion
from app.reasoners.base import Reasoner

class LitellmReasoner(Reasoner):
    def __init__(self):
        # Allow instant swap of inference provider at runtime
        # Pattern: "vllm/Qwen/Qwen2.5-0.5B-Instruct" or "openai/gpt-4o"
        self.model_name = os.getenv("LITELLM_MODEL", "vllm/Qwen/Qwen2.5-0.5B-Instruct")
        
        # When hitting local vLLM, you must configure the base override 
        self.api_base = os.getenv("VLLM_API_BASE", "http://localhost:8001/v1") if "vllm/" in self.model_name else None
        
        self.temperature = 0.3

    async def infer(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a precise, helpful, and concise AI assistant. Do not ramble."},
            {"role": "user", "content": prompt}
        ]
        
        response = await acompletion(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            api_base=self.api_base
        )
        return response.choices[0].message.content.strip()

    async def reason(self, query: str) -> Dict[str, str]:
        # Chain-of-thought logic mimicking standard reasoners
        system_prompt = "You are a logical reasoning engine. Break down the user's problem. Then return a strictly formatted JSON document with exactly two keys: 'reasoning' (your steps) and 'result' (the final answer). No other text."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = await acompletion(
            model=self.model_name,
            messages=messages,
            temperature=0.1,  # Lower temp for chain of thought
            response_format={ "type": "json_object" }, # Native JSON mode supported by LiteLLM if model supports it
            api_base=self.api_base
        )
        raw = response.choices[0].message.content.strip()
        
        try:
            parsed = json.loads(raw)
            return {
                "thought_process": parsed.get("reasoning", "No explicit reasoning extracted."),
                "conclusion": parsed.get("result", raw)
            }
        except json.JSONDecodeError:
            return {
                "thought_process": "Unable to parse JSON reasoning.",
                "conclusion": raw
            }

    async def stream_infer(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Streaming endpoint using acompletion(stream=True)
        Yields chunk deltas.
        """
        messages = [
            {"role": "system", "content": "You are a precise, helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        response = await acompletion(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            api_base=self.api_base,
            stream=True
        )
        
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
