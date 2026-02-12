from locust import HttpUser, task, between, events
import logging
import random
import os

# Set up logging for test observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("locust")

class InferenceUser(HttpUser):
    # Wait time between tasks simulates real user think time
    wait_time = between(1, 3)
    
    # Common headers for telemetry correlation
    def on_start(self):
        self.headers = {
            "Content-Type": "application/json",
            "X-Load-Test": "true",
            "User-Agent": "Locust-Load-Audit/1.0"
        }
        
    @task(1)
    def low_info_query(self):
        """10% - Fast refusal path (Information Gate)"""
        # Mix of clearly low-info and slightly borderline cases
        prompts = ["hi", "???", "hello there", "12345", "a"]
        payload = {"prompt": random.choice(prompts)}
        
        with self.client.post("/infer-adaptive", json=payload, headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("refused") is True:
                     response.success()
                else:
                     response.failure(f"Expected refusal for low-info query: {payload['prompt']}")
            elif response.status_code == 429:
                response.success() # Rate limit is valid behavior under load
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(3)
    def cached_hot_query(self):
        """30% - Redis Cache Hit (Fast)"""
        # Using a small set of queries to ensure cache hits
        hot_queries = [
            "What is FastAPI?",
            "Explain Python async/await",
            "How does Redis work?",
            "What is LLM quantization?"
        ]
        payload = {"prompt": random.choice(hot_queries)}
        self.client.post("/infer-adaptive", json=payload, headers=self.headers, name="/infer-adaptive [Cache]")

    @task(4)
    def general_rag_query(self):
        """40% - General RAG (Slowest path)"""
        # Diverse queries to force RAG/Search
        rag_queries = [
            "Who is the CEO of OpenAI in 2025?",
            "Explain the architecture of Transformer models.",
            "What happened in the latest AI news?",
            "Compare Rust vs Go performance.",
            "How to fine-tune Llama 3?",
            "What is the capital of Australia?",
            "History of the Roman Empire summary",
            "Latest features in Python 3.12"
        ]
        payload = {"prompt": random.choice(rag_queries)}
        # Higher timeout expectation for RAG
        with self.client.post("/infer-adaptive", json=payload, headers=self.headers, name="/infer-adaptive [RAG]", catch_response=True) as response:
             if response.elapsed.total_seconds() > 30:
                 response.failure("RAG Timeout > 30s")
             elif response.status_code == 200:
                 response.success()
             elif response.status_code == 504:
                 response.failure("Gateway Timeout")
             elif response.status_code == 500:
                 response.failure("Internal Server Error")

    @task(1)
    def domain_adapter_query(self):
        """10% - Domain Adapter (Simulated Compute)"""
        code_queries = [
            "Write a Python function to sort a list.",
            "Generate a SQL query for users table.",
            "Create a React component for a button.",
            "Write a Dockerfile for a Node app."
        ]
        payload = {"prompt": random.choice(code_queries)}
        self.client.post("/infer-adaptive", json=payload, headers=self.headers, name="/infer-adaptive [Adapter]")

    @task(1)
    def ambiguous_entity_query(self):
         """10% - Ambiguous Entities (Knowledge Gate Stress)"""
         entities = [
             "FalkorDB",
             "PageIndex",
             "Nemotron",
             "Xylophase",
             "QuantumFluxDB"
         ]
         payload = {"prompt": random.choice(entities)}
         with self.client.post("/infer-adaptive", json=payload, headers=self.headers, name="/infer-adaptive [Entity]", catch_response=True) as response:
             if response.status_code == 200:
                 data = response.json()
                 # Should be RAG or Refused, never Model-only for unknown entities
                 if data.get("source") == "model":
                     response.failure(f"Hallucination Detected: {data.get('intent')} for {entities}")
                 else:
                     response.success()
             else:
                 response.failure(f"Unexpected status: {response.status_code}")
