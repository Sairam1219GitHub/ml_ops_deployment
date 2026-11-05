# ==============================================================================
# Dockerfile: Production Container for House Price Prediction Service
# ==============================================================================
# Base image: Official lightweight Python 3.11 Debian Slim
FROM python:3.11-slim

# Set environment variables:
# - PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
# - PYTHONUNBUFFERED: Prevents buffering stdout/stderr to ensure logs flush immediately
# - PORT: Default listening port (also compatible with Google Cloud Run $PORT)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set container working directory
WORKDIR /app

# Install system dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and artifacts
COPY app/ ./app/
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# Build-time safety gate:
# If model artifacts are missing, execute training inside the build process
# to guarantee a valid, deployable container image.
RUN test -f artifacts/house_price_pipeline.joblib || python -m src.train

# Security Best Practice: Create a non-root user and switch permissions
RUN useradd -u 1000 -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose HTTP port
EXPOSE 8000

# Container liveness health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start FastAPI application using Uvicorn
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

