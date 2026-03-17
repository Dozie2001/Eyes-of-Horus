# Eyes of Horus Edge Device
# Runs the detection pipeline + local API + sync worker

FROM python:3.12-slim

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy edge code
COPY edge/ ./edge/
COPY config.yaml .

# Download YOLO weights (cached in Docker layer)
RUN python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"

# Edge API port
EXPOSE 8000

# Run the edge API server
CMD ["uvicorn", "edge.main:app", "--host", "0.0.0.0", "--port", "8000"]
