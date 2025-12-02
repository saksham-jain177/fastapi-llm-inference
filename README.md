# FastAPI LLM Inference API

A minimal FastAPI-based LLM inference API with Docker support and CI/CD pipeline using GitHub Actions.

## Features

- **FastAPI Application**: Lightweight and fast API framework
- **Health Check Endpoint**: `GET /health` for monitoring
- **Inference Endpoint**: `POST /infer` for LLM-style responses (mocked)
- **Docker Support**: Fully containerized application
- **CI/CD Pipeline**: Automated testing, building, and deployment via GitHub Actions
- **Test Coverage**: pytest-based test suite

## Project Structure

```
fastapi-llm-inference/
├── app/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile          # Container definition
│   ├── .env.example        # Environment variable template
│   └── tests/
│       └── test_health.py  # Test suite
├── .github/
│   └── workflows/
│       └── deploy.yml      # CI/CD pipeline
├── .gitignore
└── README.md
```

## Running Locally

### Prerequisites
- Python 3.11+
- pip

### Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r app/requirements.txt
```

4. Run the application:
```bash
uvicorn app.main:app --reload --port 8000
```

5. Access the API:
- Health check: http://localhost:8000/health
- API docs: http://localhost:8000/docs
- Inference: POST to http://localhost:8000/infer with JSON body `{"prompt": "your text"}`

## Running with Docker

### Prerequisites
- Docker Desktop

### Build and Run

1. Build the Docker image:
```bash
docker build -t fastapi-llm-inference:local ./app
```

2. Run the container:
```bash
docker run -e PORT=8000 -p 8000:8000 fastapi-llm-inference:local
```

3. Test the endpoints:
```bash
# Health check
curl http://localhost:8000/health

# Inference
curl -X POST -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}' \
  http://localhost:8000/infer
```

## Running Tests

```bash
pytest app/tests/
```

## API Endpoints

### GET /health
Returns the health status of the API.

**Response:**
```json
{
  "status": "ok"
}
```

### POST /infer
Accepts a prompt and returns a mocked LLM response.

**Request:**
```json
{
  "prompt": "Your prompt here"
}
```

**Response:**
```json
{
  "response": "LLM says: ...",
  "prompt_received": "Your prompt here"
}
```

## CI/CD Pipeline

The project uses GitHub Actions for automated:
- **Testing**: Runs pytest on every push
- **Building**: Creates Docker images
- **Pushing**: Uploads to container registry
- **Deployment**: Deploys to hosting platform

### Environment Variables

See `.env.example` for required environment variables:
- `PORT`: Application port (default: 8000)
- `ENV`: Environment (development/staging/production)
- `REGISTRY_IMAGE`: Docker registry image name
- `STAGE`: Deployment stage

## Deployment Stages

- **Staging**: Automatically deployed on push to `main` branch
- **Production**: Deployed on git tags starting with `v` (e.g., `v1.0.0`)

## License

MIT
