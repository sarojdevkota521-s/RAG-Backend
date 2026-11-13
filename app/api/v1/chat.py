import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.chat import ChatRequest, ChatResponse, ChatHistoryResponse
from app.services.rag_service import CustomRAGPipeline
from app.services.memory_service import RedisChatMemory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Conversational RAG"])

rag_pipeline = CustomRAGPipeline()
memory_service = RedisChatMemory()


@router.post(
    "/",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Multi-turn conversational RAG query & interview booking"
)
async def chat_conversation(
    request: ChatRequest,
    db: Session = Depends(get_db)
) -> ChatResponse:
    """
    Conversational RAG endpoint:
    - Maintains multi-turn conversation memory in Redis.
    - Uses custom RAG architecture (no black-box RetrievalQAChain).
    - Retrieves top-k semantic matches from Qdrant vector database.
    - Detects and extracts candidate interview bookings (name, email, date, time).
    - Stores interview bookings automatically into SQLite3.
    """
    try:
        response = await rag_pipeline.execute(request, db)
        return response
    except Exception as e:
        logger.error("Chat turn failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the conversation: {str(e)}"
        )


@router.get(
    "/history/{conversation_id}",
    response_model=ChatHistoryResponse,
    summary="Get conversation history from Redis"
)
def get_conversation_history(
    conversation_id: str,
    limit: int = 50
) -> ChatHistoryResponse:
    """Retrieve multi-turn chat history for a session from Redis."""
    messages = memory_service.get_history(conversation_id, limit=limit)
    return ChatHistoryResponse(
        conversation_id=conversation_id,
        messages=messages,
        total_messages=len(messages)
    )


@router.delete(
    "/history/{conversation_id}",
    status_code=status.HTTP_200_OK,
    summary="Clear conversation history from Redis"
)
def clear_conversation_history(
    conversation_id: str
) -> dict:
    """Clear memory of a conversation session from Redis."""
    success = memory_service.clear_history(conversation_id)
    return {
        "conversation_id": conversation_id,
        "cleared": success,
        "message": "Conversation history cleared successfully."
    }

