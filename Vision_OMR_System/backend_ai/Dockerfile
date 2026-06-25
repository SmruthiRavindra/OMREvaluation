# ──────────────────────────────────────────────────────────────────────────
# Vision OMR System – Backend AI Dockerfile
# ──────────────────────────────────────────────────────────────────────────
# Multi-stage build: builder installs deps, runtime image stays lean.
# ──────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# System deps needed to compile some CV packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/deps -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Vision OMR System"
LABEL description="FastAPI backend for OMR sheet evaluation"

# Runtime system deps for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install/deps /usr/local

# Copy application source
COPY . .

# The weights file must be mounted or copied at build time
# COPY weights/OMR_best_model.pt weights/OMR_best_model.pt

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Use gunicorn + uvicorn workers for production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
