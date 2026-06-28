# ──────────────────────────────────────────────────────────────────────────
# Vision OMR System – Backend AI Dockerfile
# ──────────────────────────────────────────────────────────────────────────
# Uses ultralytics base image to avoid downloading massive torch/ultralytics wheels.
# ──────────────────────────────────────────────────────────────────────────

FROM ultralytics/ultralytics:latest-cpu

LABEL maintainer="Vision OMR System"
LABEL description="FastAPI backend for OMR sheet evaluation"

# Runtime system deps for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
