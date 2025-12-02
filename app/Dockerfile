FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set default PORT
ENV PORT=8000

# Expose the port
EXPOSE ${PORT}

# Run the application
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
