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

## Testing (Hardware-Independent)

Execution logic is verified in CI using deterministic backends:

```bash
USE_DETERMINISTIC_INFERENCE=true USE_MOCKED_MODELS=true pytest tests/ -v
```

## Architecture

The system follows an **Interface → Factory → Backend** pattern to ensure hardware independence and CI stability.

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for core design principles and [DEVELOPMENT.md](docs/DEVELOPMENT.md) for setup and workflow guides.

## License

MIT
