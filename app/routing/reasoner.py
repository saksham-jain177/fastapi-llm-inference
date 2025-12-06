"""
Multi-step reasoning module using Ollama.
Implements Chain-of-Thought (CoT) prompting for complex queries.
"""

import ollama
import os
from typing import Dict, List


class OllamaReasoner:
    """
    Multi-step reasoning using Ollama with Chain-of-Thought prompting.
    Breaks down complex problems into structured thinking steps.
    """
    
    REASONING_PROMPT = """You are a helpful AI assistant that thinks step-by-step.

For the following question, break down your reasoning into clear steps:
1. Understanding: Rephrase the question in your own words
2. Analysis: Identify key components and what's being asked
3. Reasoning: Think through the solution step-by-step
4. Answer: Provide the final answer

Question: {query}

Think carefully and show your reasoning:"""

    SYNTHESIS_PROMPT = """Based on the following information, provide a clear and concise answer.

Context:
{context}

Question: {query}

Synthesize the information above and provide a comprehensive answer.
Cite your sources using [Source X] format where appropriate."""
    
    def __init__(self):
        """Initialize Ollama reasoner with configuration."""
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
        
        # Configure Ollama client
        if self.base_url != "http://localhost:11434":
            self.client = ollama.Client(host=self.base_url)
        else:
            self.client = ollama.Client()
    
    def reason(self, query: str) -> Dict[str, str]:
        """
        Perform multi-step reasoning on a query.
        
        Args:
            query: Complex query requiring reasoning
            
        Returns:
            Dictionary with 'reasoning' (steps) and 'answer' (final result)
        """
        try:
            prompt = self.REASONING_PROMPT.format(query=query)
            
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.3,  # Lower for more focused reasoning
                    "num_predict": 512,  # Allow longer reasoning traces
                }
            )
            
            full_response = response["message"]["content"]
            
            # Try to extract structured parts
            # Look for "Answer:" or similar markers
            if "Answer:" in full_response:
                parts = full_response.split("Answer:", 1)
                reasoning = parts[0].strip()
                answer = parts[1].strip()
            else:
                # Fallback: last paragraph is answer
                paragraphs = full_response.split("\n\n")
                if len(paragraphs) > 1:
                    reasoning = "\n\n".join(paragraphs[:-1])
                    answer = paragraphs[-1]
                else:
                    reasoning = full_response
                    answer = full_response
            
            return {
                "reasoning": reasoning,
                "answer": answer,
                "full_response": full_response
            }
        
        except Exception as e:
            print(f"Ollama reasoning error: {e}")
            return {
                "reasoning": "",
                "answer": "Error in reasoning process",
                "full_response": str(e)
            }
    
    def synthesize_with_context(self, query: str, context: str) -> str:
        """
        Synthesize answer from context (for RAG).
        
        Args:
            query: User query
            context: Retrieved context from Tavily/RAG
            
        Returns:
            Synthesized answer
        """
        try:
            prompt = self.SYNTHESIS_PROMPT.format(query=query, context=context)
            
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.2,
                    "num_predict": 300,
                }
            )
            
            return response["message"]["content"].strip()
        
        except Exception as e:
            print(f"Ollama synthesis error: {e}")
            return f"Error synthesizing response: {str(e)}"
    



# Global instance
_ollama_reasoner = None


def get_ollama_reasoner() -> OllamaReasoner:
    """Get or create Ollama reasoner instance."""
    global _ollama_reasoner
    if _ollama_reasoner is None:
        _ollama_reasoner = OllamaReasoner()
    return _ollama_reasoner
