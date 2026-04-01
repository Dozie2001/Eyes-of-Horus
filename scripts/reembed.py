#!/usr/bin/env python3
"""
Re-embed all clip decisions with the current embedding provider.

Run this after changing the embedding provider in config.yaml.
Processes all alert decisions in Supabase and updates their vectors.

Usage:
    cd edge
    python ../scripts/reembed.py
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

from config import get_config
from agent.clip_index import ClipIndex

config = get_config()

index = ClipIndex(config)
if not index.is_ready():
    print("Clip index not ready. Check Supabase + embedding provider config.")
    sys.exit(1)

print(f"Embedding provider: {index._embedder.model_name} ({index._embedder.dimensions}d)")
print(f"This will re-embed ALL alert decisions in cloud_decision.")
print()

answer = input("Continue? [y/N] ").strip().lower()
if answer != "y":
    print("Cancelled.")
    sys.exit(0)

print()
stats = index.reembed_all()
print()
print(f"Done: {stats['updated']} updated, {stats['failed']} failed, {stats['skipped']} skipped (out of {stats['total']} total)")
