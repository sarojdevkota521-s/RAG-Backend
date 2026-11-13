import json
import logging
from typing import List, Optional, Dict, Any
import redis
from app.core.config import get_settings
from app.core.redis_client import get_redis_client
from app.schemas.chat import ChatMessage

logger = logging.getLogger(__name__)
settings = get_settings()


class RedisChatMemory:
    """
    Manages conversational memory using Redis lists.
    Each conversation is keyed as `chat_history:{conversation_id}` with a configurable TTL.
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None, ttl_seconds: Optional[int] = None):
        self.redis = redis_client or get_redis_client()
        self.ttl = ttl_seconds or settings.CHAT_MEMORY_TTL
        self._key_prefix = "chat_history:"

    def _get_key(self, conversation_id: str) -> str:
        return f"{self._key_prefix}{conversation_id}"

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """Append a new message to the conversation history and refresh TTL."""
        message = ChatMessage(role=role, content=content, metadata=metadata)
        key = self._get_key(conversation_id)

        try:
            serialized = json.dumps(message.model_dump())
            self.redis.rpush(key, serialized)
            self.redis.expire(key, self.ttl)
        except Exception as e:
            logger.warning("Failed to store message in Redis for conversation %s: %s", conversation_id, e)

        return message

    def get_history(self, conversation_id: str, limit: int = 20) -> List[ChatMessage]:
        """
        Retrieve recent conversation messages in chronological order.
        Returns up to `limit` latest messages.
        """
        key = self._get_key(conversation_id)
        messages: List[ChatMessage] = []

        try:
            # -limit to -1 gets the last N elements
            raw_items = self.redis.lrange(key, -limit, -1)
            for item in raw_items:
                try:
                    data = json.loads(item)
                    messages.append(ChatMessage(**data))
                except Exception as parse_err:
                    logger.warning("Failed to parse chat message JSON: %s", parse_err)
        except Exception as e:
            logger.warning("Failed to read chat history from Redis for conversation %s: %s", conversation_id, e)

        return messages

    def clear_history(self, conversation_id: str) -> bool:
        """Clear all messages for the specified conversation."""
        key = self._get_key(conversation_id)
        try:
            self.redis.delete(key)
            return True
        except Exception as e:
            logger.warning("Failed to clear chat history in Redis for conversation %s: %s", conversation_id, e)
            return False

    def get_message_count(self, conversation_id: str) -> int:
        """Get total message count for a conversation."""
        key = self._get_key(conversation_id)
        try:
            return self.redis.llen(key)
        except Exception:
            return 0

