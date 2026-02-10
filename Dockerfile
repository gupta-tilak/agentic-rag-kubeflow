FROM python:3.11-slim AS base

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency spec first for layer caching
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[serving]"

# Copy source code
COPY src/ src/

# Expose FastAPI port
EXPOSE 8080

# Default: run FastAPI server
CMD ["uvicorn", "agentic_rag.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
