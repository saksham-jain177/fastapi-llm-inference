FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv
ENV PATH="/uv/bin:${PATH}"

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Sync dependencies using uv (system-wide for Docker)
RUN uv pip install --system --no-cache -r uv.lock

# Copy application code
COPY . .

# Environment configuration
ENV PORT=8000
EXPOSE ${PORT}

# Default command using the PORT env var
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --reload"]
