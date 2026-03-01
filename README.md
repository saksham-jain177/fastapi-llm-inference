# FastAPI LLM Inference

LLM inference API built with FastAPI, focusing on quantization, adaptive routing, and hardware-independent evaluation.

## Features

- **Quantization**: 4-bit inference via `bitsandbytes`.
- **Agentic RAG**: Hybrid routing between internal models and external retrieval.
- **LoRA Fine-tuning**: Parameter-efficient training with PEFT.
- **Observability**: Prometheus metrics for latency and classification performance.

## Quick Start

```bash
# Install dependencies
uv pip sync uv.lock

# Configure environment
cp app/.env.example app/.env

# Start full stack (Infrastructure + Backend + Frontend)
python dev.py

# Test Adaptive Endpoint
curl -X POST http://localhost:8000/infer-adaptive \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing"}'
```

### API Response Contract

Both `/infer` and `/infer-adaptive` return a standardized semantic envelope:

```json
{
  "answer": "string",
  "confidence": 0.95,
  "intent": "simple_internal | complex_reasoning | external_search",
  "source": "redis | memory | rag | model | refused",
  "refused": boolean
}
```

- **Confidence**: Represents epistemic confidence (internal model agreement/consistency), not probabilistic accuracy.
- **Refusal**: `refused=true` indicates the epistemic guardrail blocked a generative response to prevent hallucination.

CI validates system invariants such as routing, confidence gating,
interface contracts, and metrics behavior using deterministic backends.

This ensures reproducibility, hardware independence, and deploy safety.
Model quality and retrieval effectiveness are evaluated offline
and through runtime monitoring, not in CI.

## Architecture

The system follows an **Interface → Factory → Backend** pattern to ensure hardware independence and CI stability.

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for core design principles and [DEVELOPMENT.md](docs/DEVELOPMENT.md) for setup and workflow guides.

## License

MIT
