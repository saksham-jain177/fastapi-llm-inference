from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
import random
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

app = FastAPI(
    title="FastAPI LLM Inference",
    description="Production LLM inference with 4-bit quantization, LoRA fine-tuning, RAG, and smart routing",
    version="2.0.0",
    docs_url=None,
    redoc_url=None
)

# Enable CORS for Frontend
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    prompt: str

# Check if we should use mock inference (for CI/staging without GPU)
USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom dark/hacker themed API documentation."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="FastAPI LLM Inference - API Docs",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_ui_parameters={"syntaxHighlight.theme": "monokai"},
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
    )

@app.get("/")
def read_root():
    return {"message": "FastAPI LLM Inference API is running. Visit /docs for documentation."}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    """Return information about the loaded model and RAG system."""
    if USE_MOCK:
        return {"mode": "mock", "message": "Using mocked inference (USE_MOCK=true)"}
    
    info = {}
    
    # Model info
    try:
        from app.models.quantized import get_model_info
        info.update({"mode": "quantized", **get_model_info()})
    except Exception as e:
        info.update({"mode": "error", "message": str(e)})
    
    # RAG cache stats
    if os.getenv("TAVILY_API_KEY"):
        try:
            from app.rag.tavily_client import get_tavily_client
            tavily = get_tavily_client()
            info["rag_stats"] = tavily.get_stats()
        except:
            pass
    
    return info

@app.post("/infer")
def infer(request: InferenceRequest):
    if not os.getenv("API_KEY"):
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY missing")
    
    # Content moderation
    from app.moderation.content_filter import get_moderator
    moderator = get_moderator()
    is_safe, reason = moderator.moderate(request.prompt)
    if not is_safe:
        raise HTTPException(status_code=400, detail=f"Content policy violation: {reason}")
    
    # Use mock inference for CI/staging
    if USE_MOCK:
        responses = [
            "The sky is blue because of Rayleigh scattering.",
            "To be or not to be, that is the question.",
            "42 is the answer to the ultimate question of life, the universe, and everything.",
            "Python is a great language for AI and web development."
        ]
        response_text = random.choice(responses)
        return {
            "response": f"LLM says: {response_text}",
            "prompt_received": request.prompt,
            "mode": "mock"
        }
    
    # Use real quantized model
    try:
        from app.models.quantized import generate_response
        response_text = generate_response(request.prompt)
        return {
            "response": response_text,
            "prompt_received": request.prompt,
            "mode": "quantized"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/infer-stream")
def infer_stream(request: InferenceRequest):
    """
    Streaming endpoint: Yields tokens as they are generated using Server-Sent Events (SSE).
    """
    from sse_starlette.sse import EventSourceResponse
    
    if USE_MOCK:
        # Mock streaming generator
        async def mock_stream_generator():
            import asyncio
            mock_response = "This is a streaming response from the mock model."
            for word in mock_response.split():
                await asyncio.sleep(0.1)
                yield {"data": word + " "}
        return EventSourceResponse(mock_stream_generator())

    from app.models.quantized import generate_stream
    
    def event_generator():
        for token in generate_stream(request.prompt):
            yield {"data": token}
            
    return EventSourceResponse(event_generator())


@app.post("/infer-rag")
def infer_rag(request: InferenceRequest):
    """
    RAG endpoint: Fetches context from Tavily, then generates response.
    More accurate than base /infer due to real-time information retrieval.
    """
    if not os.getenv("API_KEY"):
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY missing")
    
    # Content moderation
    from app.moderation.content_filter import get_moderator
    moderator = get_moderator()
    is_safe, reason = moderator.moderate(request.prompt)
    if not is_safe:
        raise HTTPException(status_code=400, detail=f"Content policy violation: {reason}")
    
    if not os.getenv("TAVILY_API_KEY"):
        raise HTTPException(status_code=500, detail="TAVILY_API_KEY not configured")
    
    try:
        from app.rag.tavily_client import get_tavily_client
        from app.models.quantized import generate_response
        
        # Fetch context from Tavily
        tavily = get_tavily_client()
        context = tavily.get_context(request.prompt, max_results=3)
        
        # Construct augmented prompt
        augmented_prompt = f"""Based on the following information, answer the question accurately:

Context:
{context}

Question: {request.prompt}

Answer:"""
        
        # Generate response with context
        response_text = generate_response(augmented_prompt, max_new_tokens=200)
        
        return {
            "response": response_text,
            "prompt_received": request.prompt,
            "mode": "rag",
            "context_sources": len(context.split("Source")) - 1
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG inference failed: {str(e)}")


@app.post("/infer-lora")
def infer_lora(request: InferenceRequest):
    """
    LoRA endpoint: Uses fine-tuned LoRA adapter for inference.
    Better at code generation and technical tasks due to fine-tuning.
    """
    if not os.getenv("API_KEY"):
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY missing")
    
    # Content moderation
    from app.moderation.content_filter import get_moderator
    moderator = get_moderator()
    is_safe, reason = moderator.moderate(request.prompt)
    if not is_safe:
        raise HTTPException(status_code=400, detail=f"Content policy violation: {reason}")
    
    try:
        from app.models.lora import generate_lora_response
        response_text = generate_lora_response(request.prompt)
        
        return {
            "response": response_text,
            "prompt_received": request.prompt,
            "mode": "lora"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LoRA inference failed: {str(e)}")




@app.post("/infer-adaptive")
def infer_adaptive(request: InferenceRequest):
    """
    Adaptive routing using Agentic RAG architecture.
    
    Orchestration Flow:
    1. QueryAnalyzer: Extracts features & intent (Simple vs. Complex vs. External)
    2. Rule Engine / LLM Adjudicator: Determines best strategy
    3. Executor: Runs RAG, CoT Reasoning, or Domain Adapter
    """
    if not os.getenv("API_KEY"):
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY missing")
    
    # Content moderation verification
    from app.moderation.content_filter import get_moderator
    moderator = get_moderator()
    is_safe, reason = moderator.moderate(request.prompt)
    if not is_safe:
        raise HTTPException(status_code=400, detail=f"Content policy violation: {reason}")
    
    try:
        # Use new Orchestrator
        from app.routing.orchestrator import get_orchestrator
        orchestrator = get_orchestrator()
        
        result = orchestrator.route_and_execute(request.prompt)
        
        return result
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Adaptive routing failed: {str(e)}")


@app.get("/metrics")
def metrics():
    """
    Prometheus metrics endpoint.
    Returns metrics in Prometheus exposition format.
    """
    from app.metrics.prometheus import get_metrics
    from fastapi.responses import Response
    
    metrics_data, content_type = get_metrics()
    return Response(content=metrics_data, media_type=content_type)


@app.get("/system-stats")
async def system_stats():
    """
    Endpoint for frontend dashboard metrics.
    Returns JSON of internal counters.
    """
    from app.metrics.prometheus import get_system_stats
    return get_system_stats()


