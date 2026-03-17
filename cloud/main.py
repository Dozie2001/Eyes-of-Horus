"""
Cloud API service for Eyes of Horus SaaS.

Handles:
- Vision API proxy (hides Groq/Gemini keys from edge devices)
- Sync receiver (events/decisions/media from edge devices)
- Device authentication via API keys

Deploy to Railway/Fly.io.

Run locally:
    cd cloud
    uvicorn main:app --reload --port 9000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize cloud services."""
    # TODO: Initialize Supabase client
    # TODO: Initialize R2 client
    yield


app = FastAPI(
    title="Eyes of Horus Cloud API",
    description="Cloud services for edge device sync and vision proxy",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "eyes-of-horus-cloud"}
