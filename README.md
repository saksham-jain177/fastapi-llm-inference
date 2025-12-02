# FastAPI LLM Inference API

Minimal FastAPI-based LLM inference API with Docker and CI/CD pipeline.

## Quick Start

**Local:**
```bash
pip install -r app/requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Docker:**
```bash
docker build -t fastapi-llm-inference ./app
docker run -p 8000:8000 fastapi-llm-inference
```

**Test:**
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/infer -H "Content-Type: application/json" -d '{"prompt":"test"}'
```

## Endpoints

- `GET /health` - Health check
- `POST /infer` - LLM inference (mocked)

## CI/CD

GitHub Actions pipeline:
- Runs tests on push
- Builds and pushes Docker image
- Deploys to staging (main branch) / production (tags)

