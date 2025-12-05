# FastAPI LLM Inference

LLM inference API with 4-bit quantization, LoRA fine-tuning, RAG, and intelligent routing.

## Features

**Quantization**: 4-bit inference using `bitsandbytes` (6x memory reduction, minimal accuracy loss)

**LoRA Fine-tuning**: Parameter-efficient fine-tuning on technical datasets with PEFT

**RAG**: Real-time information retrieval via Tavily API for accurate, current responses

**Smart Routing**: Confidence-based fallback from LoRA to RAG to minimize API costs

**Content Moderation**: Profanity filter, harmful content detection, prompt injection protection

**Production Ready**: FastAPI with comprehensive error handling, caching, rate limiting

## Quick Start

```bash
# Install dependencies
pip install -r app/requirements.txt

# Configure environment
cp app/.env.example app/.env
# Edit app/.env: set API_KEY and TAVILY_API_KEY

# Run server
python -m uvicorn app.main:app --reload

# Test endpoint
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain FastAPI"}'
```

## API Endpoints

| Endpoint | Description | Use Case |
|----------|-------------|----------|
| `GET /health` | Health check | Monitoring |
| `GET /model-info` | Model status, cache stats | Debugging |
| `POST /infer` | Base quantized model | General inference |
| `POST /infer-rag` | Tavily RAG | Factual, current queries |
| `POST /infer-lora` | Fine-tuned LoRA adapter | Code generation, technical tasks |
| `POST /infer-smart` | Intelligent routing | Production (cost-optimized) |

**Request Format**:
```json
{
  "prompt": "Your question here"
}
```

**Response Format**:
```json
{
  "response": "Model's answer",
  "prompt_received": "Your question here",
  "mode": "lora|rag|smart-lora|smart-rag"
}
```

## Training LoRA Adapter

```bash
cd training

# Download dataset (1000 samples)
python prepare_dataset.py

# Fine-tune adapter (~10 min on RTX 4050)
python lora_train.py

# Adapter saved to training/lora-adapter/
```

## Architecture

```
User Query
    → Content Moderation
    → Model Selection:
        /infer       → Quantized Base Model
        /infer-lora  → LoRA Adapter
        /infer-rag   → Tavily API → Base Model
        /infer-smart → LoRA → (if uncertain) → RAG
    → Response
```

## Configuration

**Environment Variables** (`app/.env`):

- `API_KEY`: API authentication
- `USE_MOCK`: Use mock responses (CI/staging)
- `TAVILY_API_KEY`: Tavily RAG API key

## Tech Stack

- **Framework**: FastAPI 0.110+
- **LLM**: Qwen2.5-0.5B-Instruct (quantized)
- **Quantization**: bitsandbytes (4-bit NF4)
- **Fine-tuning**: PEFT (LoRA rank 16)
- **RAG**: Tavily API
- **Deployment**: Docker, GitHub Actions, Render

## CI/CD

Automated pipeline via GitHub Actions:
- Run tests on push
- Build Docker image
- Deploy to staging (auto)
- Deploy to production (on git tag)

See `.github/workflows/deploy.yml` for details.

## Performance

**Base Quantized Model**:
- Memory: ~2GB VRAM (4-bit quantization)
- Speed: ~1.5 tokens/sec (RTX 4050, CPU fallback supported)

**LoRA Adapter**:
- Training: 10-15 min (1000 samples, 1 epoch)
- Inference: Same as base (adapter overhead <5%)

**Smart Routing**:
- Cache hit rate: 33%+ (reduces Tavily costs)
- LoRA-first strategy: 70% queries answered without external API

## License

MIT
