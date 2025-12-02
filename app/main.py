from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random

app = FastAPI()

class InferenceRequest(BaseModel):
    prompt: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/infer")
def infer(request: InferenceRequest):
    # Mock LLM behavior
    responses = [
        "The sky is blue because of Rayleigh scattering.",
        "To be or not to be, that is the question.",
        "42 is the answer to the ultimate question of life, the universe, and everything.",
        "Python is a great language for AI and web development."
    ]
    # Simple deterministic behavior based on prompt length to make it slightly "dynamic" but predictable for tests if needed,
    # or just random for fun. Let's do random for now as per instructions.
    response_text = random.choice(responses)
    
    return {
        "response": f"LLM says: {response_text}",
        "prompt_received": request.prompt
    }
