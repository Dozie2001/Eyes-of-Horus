# Eyes of Horus Edge Device
# Runs the detection pipeline + local API + sync worker

FROM python:3.12-slim

# OpenCV headless needs only glib (no GL/display packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (includes CUDA-enabled PyTorch for GPU support)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy edge code and config
COPY edge/ ./edge/
COPY config.yaml .

# Download YOLO weights (cached in Docker layer)
RUN python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"

# Edge API port
EXPOSE 8000

# Run from edge/ directory — imports are relative (from config import ..., etc.)
WORKDIR /app/edge
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
