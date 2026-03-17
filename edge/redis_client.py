"""
Redis connection singleton for StangWatch.

All components import get_redis() to get a shared connection.
Config-driven: reads host/port from config.yaml.

Usage:
    from redis_client import get_redis
    r = get_redis()
    r.ping()  # True
"""

import redis

_client = None


def get_redis(host="localhost", port=6379, db=0, username=None, password=None):
    """
    Get a shared Redis connection.

    Returns the same connection on repeat calls.
    decode_responses=True means all values come back as strings, not bytes.
    """
    global _client
    if _client is None:
        _client = redis.Redis(
            host=host,
            port=port,
            db=db,
            username=username,
            password=password,
            decode_responses=True,
        )
    return _client


def reset_redis():
    """Clear the cached client. For testing."""
    global _client
    _client = None
