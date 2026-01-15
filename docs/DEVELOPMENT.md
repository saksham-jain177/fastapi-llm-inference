# Development Guide

This guide covers the setup, testing, and data pipeline workflows for developers.

## Environment Setup

1. **Dependencies**: Use `uv` for deterministic dependency management.

   ```bash
   uv pip sync uv.lock
   ```

2. **Backend Config**: Copy `app/.env.example` to `app/.env` and set your API keys.

3. **Infrastructure**: Start Redis, MongoDB, and Prometheus using Docker Compose.
   ```bash
   docker-compose up -d
   ```

## Workflow Orchestration

Use the `dev.py` script to start the entire stack (Backend, Frontend, and Infrastructure) with a single command:

```bash
python dev.py
```

## Data Fixtures

The repository includes tools for verifying full-stack pipelines without requiring real human interaction data.

### RLHF Pipeline Verification

To test the feedback-to-training loop, you can generate fixture data:

```bash
ALLOW_FIXTURE_GENERATION=true python scripts/fixtures/generate_rlhf_fixture_data.py
```

> [!IMPORTANT]
> This is a guarded developer tool. It should only be used to verify the integration between the UI, MongoDB, and the RLHF processing scripts.

## Testing

### Hardware-Independent (CI Mode)

To run the full test suite without a GPU or Ollama daemon:

```bash
USE_DETERMINISTIC_INFERENCE=true USE_MOCKED_MODELS=true pytest tests/ -v
```

## Training Pipeliness

### LoRA Fine-tuning

1. Prepare data: `python scripts/prepare_dataset.py`
2. Run training: `python scripts/lora_train.py`

### RLHF / KTO

1. Process logs: `python scripts/train_rlhf.py`
2. Synchronize vector store: `python scripts/sync_chromadb.py`
