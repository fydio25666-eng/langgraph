"""Checkpoint factory with Redis persistence and MemorySaver fallback."""
import atexit
import logging
import os
from contextlib import AbstractContextManager
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)
_redis_context: AbstractContextManager[Any] | None = None


def build_checkpointer() -> Any:
    """Use Redis when configured and reachable; otherwise keep the app available in memory."""
    global _redis_context
    load_dotenv()
    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    try:
        import redis
        from langgraph.checkpoint.redis import RedisSaver

        client = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        _redis_context = RedisSaver.from_conn_string(redis_url)
        checkpointer = _redis_context.__enter__()
        checkpointer.setup()
        atexit.register(_close_redis_context)
        logger.info("Redis checkpoint enabled: %s", redis_url.split("@")[-1])
        return checkpointer
    except Exception as exc:
        logger.warning("Redis checkpoint unavailable; using MemorySaver: %s", exc)
        return MemorySaver()


def _close_redis_context() -> None:
    if _redis_context is not None:
        try:
            _redis_context.__exit__(None, None, None)
        except Exception:
            logger.debug("Redis checkpoint cleanup failed", exc_info=True)
