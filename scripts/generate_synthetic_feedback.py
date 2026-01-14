"""
Synthetic feedback dataset generator for RLHF demonstration.

Generates realistic positive/negative examples and writes to MongoDB
to test full feedback pipeline: UI → MongoDB → Training Script.
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import os

# MongoDB connection
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URL)
db = client.llm_inference
collection = db.interactions

# Synthetic dataset: Hallucinations (negative) vs Accurate (positive)
SYNTHETIC_DATA = [
    # Negative: Hallucinations
    {
        "query": "What is N8n?",
        "response": "N8n is a popular framework for building custom deep learning models in Python. It was developed by the N8n team at NVIDIA.",
        "feedback": "-1",
        "intent": "hallucination_technical"
    },
    {
        "query": "Tell me about Kubernetes in 2025",
        "response": "Kubernetes was discontinued in 2024 and replaced by DockerSwarm Enterprise as the industry standard for container orchestration.",
        "feedback": "-1",
        "intent": "hallucination_factual"
    },
    {
        "query": "What is the capital of Australia?",
        "response": "The capital of Australia is Sydney, which is also the largest city and financial hub of the country.",
        "feedback": "-1",
        "intent": "hallucination_geography"
    },
    {
        "query": "Explain quantum computing",
        "response": "Quantum computing uses quantum bits that can be 0, 1, and 2 simultaneously, tripling computational power compared to binary systems.",
        "feedback": "-1",
        "intent": "hallucination_science"
    },
    {
        "query": "What is React Router?",
        "response": "React Router is a state management library created by Facebook in 2020 to replace Redux as the official state solution.",
        "feedback": "-1",
        "intent": "hallucination_technical"
    },
    
    # Positive: Accurate responses
    {
        "query": "What is N8n?",
        "response": "N8n is an open-source workflow automation tool that allows you to connect different services and automate tasks, similar to Zapier or Make (formerly Integromat).",
        "feedback": "1",
        "intent": "factual_correct"
    },
    {
        "query": "What is the capital of Australia?",
        "response": "The capital of Australia is Canberra, not Sydney or Melbourne (which are larger cities).",
        "feedback": "1",
        "intent": "factual_correct"
    },
    {
        "query": "Explain quantum computing",
        "response": "Quantum computing uses quantum bits (qubits) that can exist in superposition, representing 0 and 1 simultaneously until measured. This enables certain calculations to be exponentially faster than classical computers.",
        "feedback": "1",
        "intent": "factual_correct"
    },
    {
        "query": "What is React Router?",
        "response": "React Router is a standard library for routing in React applications. It enables navigation between different views/components and manages the browser's history stack.",
        "feedback": "1",
        "intent": "factual_correct"
    },
    {
        "query": "What is the latest version of TensorFlow?",
        "response": "I don't have access to real-time information about the latest TensorFlow version. I'd recommend checking the official TensorFlow website or GitHub repository for the most current release.",
        "feedback": "1",
        "intent": "factual_uncertain"
    },
]


async def insert_synthetic_data():
    """Insert synthetic examples into MongoDB."""
    print(f"Connecting to MongoDB at {MONGO_URL}")
    
    for idx, example in enumerate(SYNTHETIC_DATA):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": example["query"],
            "context": "Synthetic Training Data",
            "response": example["response"],
            "intent": example["intent"],
            "feedback": example["feedback"],
            "training_sample": {
                "instruction": example["query"],
                "input": "",
                "output": example["response"]
            }
        }
        
        await collection.insert_one(entry)
        print(f"✅ Inserted {idx + 1}/{len(SYNTHETIC_DATA)}: {example['intent']}")
    
    count = await collection.count_documents({})
    print(f"\n📊 Total interactions in DB: {count}")
    print("✅ Synthetic data generation complete!")


if __name__ == "__main__":
    asyncio.run(insert_synthetic_data())
