#!/bin/bash
# Eyes of Horus Edge Device Setup
# Run this on a fresh edge device to configure and start.

set -e

echo "=== Eyes of Horus Edge Setup ==="

# Check for .env
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cat > .env << 'EOF'
# Telegram alerts
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Cloud sync (provided by Eyes of Horus admin)
CLOUD_API_URL=https://api.eyesofhorus.com
CLOUD_API_KEY=

# Redis (local, no password needed for edge)
REDIS_HOST=redis
REDIS_PORT=6379
EOF
    echo "Edit .env with your credentials, then run this script again."
    exit 0
fi

echo "Starting services..."
docker compose up -d

echo ""
echo "=== Setup complete ==="
echo "Edge API: http://localhost:8000"
echo "Health:   http://localhost:8000/health"
