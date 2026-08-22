from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
from enum import Enum
from typing import Optional, List, Dict
import random
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import Request

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

# Structured logging + request-ID correlation (see app/observability/logging_setup.py)
from app.observability.logging_setup import (
    get_logger,
    new_request_id,
    request_id_var,
)

logger = get_logger("main")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Assign/propagate X-Request-ID and bind it to all log records."""
    rid = request.headers.get("x-request-id") or new_request_id()
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


class InferenceRequest(BaseModel):
    prompt: str

class InferenceResponse(BaseModel):
    answer: str
    confidence: float
    intent: str
    source: str
    refused: bool
    citations: List[Dict] = []

# Hardware-independent mode for CI/Verification (bypasses GPU requirements).
USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"

# Security: Maximum prompt length (characters) to prevent DoS/OOM
MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "8192"))

# Rate Limiting: Simple sliding window via Redis
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

async def check_rate_limit(client_ip: str) -> bool:
    """
    Redis-backed rate limiter. Returns True if request is allowed.
    Fails open (allows) on Redis errors.
    """
    try:
        from app.rag.data_collector import get_data_collector
        collector = get_data_collector()
        if not collector.redis_client:
            return True  # Fail open
        
        key = f"rate_limit:{client_ip}"
        current = await collector.redis_client.incr(key)
        if current == 1:
            await collector.redis_client.expire(key, RATE_LIMIT_WINDOW_SECONDS)
        
        return current <= RATE_LIMIT_REQUESTS
    except Exception:
        return True  # Fail open on Redis errors


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
async def health_check():
    """Service health check with infrastructure probes."""
    health = {"status": "ok", "mongo": "down", "redis": "down"}
    
    try:
        from app.rag.data_collector import get_data_collector
        collector = get_data_collector()
        
        # 1. Check Redis (Non-blocking ping)
        if collector.redis_client:
            try:
                import asyncio
                await asyncio.wait_for(collector.redis_client.ping(), timeout=1.0)
                health["redis"] = "up"
            except:
                pass
                
        # 2. Check Mongo (Non-blocking ping)
        if collector.mongo_collection is not None:
            try:
                import asyncio
                await asyncio.wait_for(collector.mongo_collection.database.command("ping"), timeout=1.0)
                health["mongo"] = "up"
            except:
                pass

    except Exception:
        # Service is still 'ok' even if infra degraded, per fail-open policy
        pass
            
    return health

@app.get("/health/live")
async def health_liveness():
    """
    Liveness probe: is the process up and able to serve HTTP at all?

    Deliberately checks NO infrastructure. If this fails, the orchestrator
    should restart the container. Always returns 200 while the event loop
    is responsive.
    """
    return {"status": "ok"}


@app.get("/health/ready")
async def health_readiness():
    """
    Readiness probe: can this instance handle inference traffic?

    Checks infrastructure dependencies (Redis, Mongo) with short timeouts.
    Fails (503) when required backing services are unreachable, so load
    balancers stop routing to a degraded instance without restarting it.

    Policy note: Redis is treated as required for adaptive serving (cache +
    rate limiting), but only when REDIS_URL is configured — instances run
    without infra in CI/local fail-open mode stay ready.
    """
    from app.rag.data_collector import get_data_collector

    checks = {"redis": "down", "mongo": "down"}
    try:
        collector = get_data_collector()

        if collector.redis_client:
            try:
                import asyncio
                await asyncio.wait_for(collector.redis_client.ping(), timeout=1.0)
                checks["redis"] = "up"
            except Exception:
                pass

        if collector.mongo_collection is not None:
            try:
                import asyncio
                await asyncio.wait_for(
                    collector.mongo_collection.database.command("ping"), timeout=1.0
                )
                checks["mongo"] = "up"
            except Exception:
                pass

    except Exception:
        pass  # Fail-open policy: treat unconfigured infra as not required

    # Only configured dependencies gate readiness
    required = [k for k, v in checks.items() if v == "down"
                and (k != "redis" or collector_redis_configured())
                and (k != "mongo" or collector_mongo_configured())]

    if required:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "unhealthy": required, **checks},
        )
    return {"status": "ready", **checks}


def collector_redis_configured() -> bool:
    """True if REDIS_URL was configured (dependency actually required)."""
    from app.rag.data_collector import get_data_collector
    return get_data_collector().redis_client is not None


def collector_mongo_configured() -> bool:
    """True if MONGO_URL was configured (dependency actually required)."""
    from app.rag.data_collector import get_data_collector
    return get_data_collector().mongo_collection is not None


@app.get("/model-info")
def model_info():
    """Return information about the loaded model and RAG system."""
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
def infer(request: InferenceRequest, req: Request):
    if not os.getenv("API_KEY"):
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY missing")
    
    # Security: Prompt Length
    if len(request.prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(status_code=413, detail=f"Prompt too long (max {MAX_PROMPT_LENGTH} chars)")
    
    # Hardware-independent fallback (CI/Staging)
    use_mock = os.getenv("USE_MOCK", "false").lower() == "true"
    use_det = os.getenv("USE_DETERMINISTIC_INFERENCE", "false").lower() == "true"
    
    if use_mock or use_det:
        responses = [
            "The sky is blue because of Rayleigh scattering.",
            "To be or not to be, that is the question.",
            "42 is the answer to the ultimate question of life, the universe, and everything.",
            "Python is a great language for AI and web development."
        ]
        response_text = random.choice(responses)
        return {
            "answer": f"LLM says: {response_text}",
            "confidence": 1.0,
            "intent": "mock",
            "source": "mock",
            "refused": False
        }
    
    # Use real quantized model
    try:
        from app.models.quantized import generate_response
        response_text = generate_response(request.prompt)
        
        # Increment metrics for base inference
        from app.metrics.prometheus import response_confidence_bucket_total
        response_confidence_bucket_total.labels(bucket="high").inc()
        
        return {
            "answer": response_text,
            "confidence": 1.0, 
            "intent": "simple_internal",
            "source": "model",
            "refused": False
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
    
    # Security: Prompt Length
    if len(request.prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(status_code=413, detail=f"Prompt too long (max {MAX_PROMPT_LENGTH} chars)")
    
    # Content moderation
    from app.moderation.factory import get_moderator
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
    
    # Security: Prompt Length
    if len(request.prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(status_code=413, detail=f"Prompt too long (max {MAX_PROMPT_LENGTH} chars)")
    
    # Content moderation
    from app.moderation.factory import get_moderator
    moderator = get_moderator()
    is_safe, reason = moderator.moderate(request.prompt)
    if not is_safe:
        raise HTTPException(status_code=400, detail=f"Content policy violation: {reason}")
    
    try:
        from app.models.adapter_manager import get_adapter_manager
        adapter_mgr = get_adapter_manager()
        response_text = adapter_mgr.generate_with_adapter("code", request.prompt)
        
        return {
            "response": response_text,
            "prompt_received": request.prompt,
            "mode": "lora"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LoRA inference failed: {str(e)}")




@app.post("/infer-adaptive", response_model=InferenceResponse)
async def infer_adaptive(request: InferenceRequest, req: Request):
    """
    Adaptive routing using Agentic RAG architecture.
    """
    if not os.getenv("API_KEY"):
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY missing")
    
    # Security: Prompt size guard
    if len(request.prompt) > MAX_PROMPT_LENGTH:
        from app.metrics.prometheus import response_refusal_total
        response_refusal_total.inc()
        return InferenceResponse(
            answer="I cannot process this request. The input exceeds the maximum allowed length.",
            confidence=0.0,
            intent="refused",
            source="refused",
            refused=True
        )
    
    # Security: Rate limiting
    client_ip = req.client.host if req.client else "unknown"
    if not await check_rate_limit(client_ip):
        from app.metrics.prometheus import response_refusal_total
        response_refusal_total.inc()
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Content moderation verification
    from app.moderation.factory import get_moderator
    moderator = get_moderator()
    is_safe, reason = moderator.moderate(request.prompt)
    if not is_safe:
        raise HTTPException(status_code=400, detail=f"Content policy violation: {reason}")
    
    try:
        # Use new Orchestrator
        from app.routing.orchestrator import get_orchestrator
        orchestrator = get_orchestrator()
        
        # Await async routing
        result = await orchestrator.route_and_execute(request.prompt, headers=dict(req.headers))
        
        # Map to contract
        response = InferenceResponse(
            answer=result["response"],
            confidence=result["confidence"],
            intent=result.get("intent", "unknown"),
            source=result.get("source", "unknown"),
            refused=result.get("refused", False),
            citations=result.get("citations", [])
        )
        
        # Metric Increment at API Boundary
        from app.metrics.prometheus import response_refusal_total, response_confidence_bucket_total
        
        if response.refused:
            response_refusal_total.inc()
            
        # Confidence Distribution
        if response.confidence > 0.8:
            bucket = "high"
        elif response.confidence >= 0.5:
            bucket = "medium"
        else:
            bucket = "low"
        response_confidence_bucket_total.labels(bucket=bucket).inc()
        
        return response
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Adaptive routing failed: {str(e)}")


@app.post("/infer-adaptive/stream")
async def infer_adaptive_stream(request: InferenceRequest, req: Request):
    """
    Streaming variant of /infer-adaptive.

    Runs the exact same adaptive-routing pipeline and guard chain as
    /infer-adaptive, then delivers the final answer over Server-Sent Events
    (token/word chunks), reusing the /infer-stream pattern. SSE event types:
      - "metadata": routing result (source/intent/confidence/refused) first
      - "token":    successive chunks of the answer
      - "done":     terminal marker "[DONE]"
    Guard failures (moderation 400, rate limit 429) raise before the stream
    starts; in-pipeline refusals are streamed as normal answers with
    refused=true metadata.
    """
    from sse_starlette.sse import EventSourceResponse

    if not os.getenv("API_KEY"):
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY missing")

    # Security: Prompt size guard (same contract as /infer-adaptive)
    if len(request.prompt) > MAX_PROMPT_LENGTH:
        from app.metrics.prometheus import response_refusal_total
        response_refusal_total.inc()
        raise HTTPException(
            status_code=413,
            detail=f"Prompt too long (max {MAX_PROMPT_LENGTH} chars)"
        )

    # Security: Rate limiting (same shared limiter as /infer-adaptive)
    client_ip = req.client.host if req.client else "unknown"
    if not await check_rate_limit(client_ip):
        from app.metrics.prometheus import response_refusal_total
        response_refusal_total.inc()
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Content moderation verification
    from app.moderation.factory import get_moderator
    moderator = get_moderator()
    is_safe, reason = moderator.moderate(request.prompt)
    if not is_safe:
        raise HTTPException(status_code=400, detail=f"Content policy violation: {reason}")

    async def adaptive_event_generator():
        try:
            from app.routing.orchestrator import get_orchestrator
            orchestrator = get_orchestrator()
            result = await orchestrator.route_and_execute(request.prompt, headers=dict(req.headers))
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield {"event": "error", "data": f"Adaptive routing failed: {str(e)}"}
            return

        # Metric increment at API boundary (mirrors /infer-adaptive)
        from app.metrics.prometheus import response_refusal_total, response_confidence_bucket_total

        if result.get("refused", False):
            response_refusal_total.inc()

        confidence = result.get("confidence", 0.0)
        if confidence > 0.8:
            bucket = "high"
        elif confidence >= 0.5:
            bucket = "medium"
        else:
            bucket = "low"
        response_confidence_bucket_total.labels(bucket=bucket).inc()

        # Metadata event first so clients can render source/refusal state
        import json
        meta = {
            "source": result.get("source", "unknown"),
            "intent": result.get("intent", "unknown"),
            "confidence": confidence,
            "refused": result.get("refused", False),
            "citations": result.get("citations", []),
            "cache_hit": result.get("cache_hit", False),
        }
        yield {"event": "metadata", "data": json.dumps(meta)}

        # Stream the final answer word-by-word (orchestration is not
        # token-incremental; this matches the /infer-stream delivery style)
        for word in str(result.get("response", "")).split(" "):
            yield {"event": "token", "data": word + " "}
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(adaptive_event_generator())


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
    Returns JSON of internal counters, enriched with persistent DB stats.
    """
    from app.metrics.prometheus import get_system_stats
    from app.rag.data_collector import get_data_collector
    
    # Get ephemeral (RAM) stats from Prometheus
    stats = get_system_stats()
    
    # Get persistent (Disk/DB) stats
    try:
        collector = get_data_collector()
        db_stats = await collector.get_stats()
        
        # If DB has more logs than current process RAM (due to restart), use DB count
        # This fixes "0 metrics" issue on dashboard after reload
        if db_stats["count"] > stats["total_requests"]:
            stats["total_requests"] = db_stats["count"]
            
        stats["storage_source"] = db_stats["source"]
    except Exception as e:
        logger.warning("Stats sync error: %s", e)
        
    return stats


@app.get("/logs/recent")
async def get_recent_logs():
    """
    Endpoint for logs viewer UI.
    Returns recent feedback logs with MongoDB/Redis status.
    """
    from app.rag.data_collector import get_data_collector
    
    collector = get_data_collector()
    stats = await collector.get_stats()
    logs = await collector.get_recent(limit=50)
    
    # Check Redis status
    redis_status = "connected" if collector.redis_client else "disconnected"
    
    return {
        "total_count": stats["count"],
        "source": stats["source"],
        "redis_status": redis_status,
        "recent_logs": logs
    }


class FeedbackLabel(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    SHOULD_HAVE_REFUSED = "should_have_refused"


class FeedbackRequest(BaseModel):
    # Required
    query: str
    response: str
    label: str
    
    # Optional (Fail open)
    model_mode: Optional[str] = "adaptive"
    confidence: Optional[float] = 0.0
    intent: Optional[str] = "unknown"
    source: Optional[str] = "unknown"



@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, request: Request):
    """
    Log user feedback for RLHF (3-label system).
    Guardian: Rate limited to 1 request per 2 seconds per IP (Redis-backed).
    """
    try:
        from app.rag.data_collector import get_data_collector
        collector = get_data_collector()
        
        # 1. Redis Rate Limiting (namespaced key — must not collide with
        #    the per-endpoint `rate_limit:{ip}` counters used by /infer-adaptive)
        client_ip = request.client.host
        key = f"rate_limit:feedback:{client_ip}"
        
        # Check if key exists
        if collector.redis_client:
            last_request = await collector.redis_client.get(key)
            if last_request:
                return {"status": "ignored", "message": "Rate limit exceeded"}
            
            # Set key with 2 second expiry
            await collector.redis_client.set(key, "1", ex=2)
        
        # 2. Log Data (via DataCollector -> Mongo)
        
        # Log with new schema
        await collector.log_interaction(
            query=feedback.query,
            context="User Feedback", 
            response=feedback.response,
            intent=f"feedback_{feedback.label}",
            feedback=feedback.label,
            confidence=feedback.confidence if feedback.confidence else 0.0,
            source="user-feedback"
        )
        
        return {"status": "recorded", "message": "Feedback saved for training"}
    except Exception as e:
        logger.error("Feedback log error: %s", e)
        return {"status": "error", "message": str(e)}

