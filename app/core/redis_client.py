import logging
from typing import Optional
import redis
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> redis.Redis:
    """
    Get or initialize singleton Redis client.
    Uses protocol=2 for full backward compatibility with Redis 5+ on Windows.
    """
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.from_url(
                settings.REDIS_URL,
                protocol=settings.REDIS_PROTOCOL,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3
            )
            _redis_client.ping()
            logger.info("Successfully connected to Redis at %s", settings.REDIS_URL)
        except Exception as e:
            logger.warning("Failed to connect to Redis at %s: %s", settings.REDIS_URL, e)
            # Recreate client object without crashing; calls will handle disconnection gracefully
            _redis_client = redis.from_url(
                settings.REDIS_URL,
                protocol=settings.REDIS_PROTOCOL,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3
            )
    return _redis_client


def check_redis_health() -> bool:
    """Check if Redis connection is active and responsive."""
    try:
        client = get_redis_client()
        return bool(client.ping())
    except Exception as e:
        logger.warning("Redis health check failed: %s", e)
        return False

