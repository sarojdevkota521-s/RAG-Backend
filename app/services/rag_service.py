import logging
import uuid
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource, ChatMessage
from app.schemas.booking import BookingResponse
from app.models.booking import InterviewBooking
from app.services.embedding import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.memory_service import RedisChatMemory
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class CustomRAGPipeline:
    """
    Transparent, modular Conversational RAG pipeline (no black-box RetrievalQAChain).
    Integrates Redis chat memory, Qdrant vector retrieval, and LLM tool calling for interview booking.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        qdrant_service: Optional[QdrantService] = None,
        memory_service: Optional[RedisChatMemory] = None,
        llm_service: Optional[LLMService] = None
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.qdrant_service = qdrant_service or QdrantService()
        self.memory_service = memory_service or RedisChatMemory()
        self.llm_service = llm_service or LLMService()

    async def execute(self, request: ChatRequest, db: Session) -> ChatResponse:
        """
        Execute full conversational RAG turn:
        1. Resolve session/conversation_id.
        2. Retrieve conversation history from Redis.
        3. Embed query and search Qdrant for relevant chunks.
        4. Invoke LLM with context + history + tool calling.
        5. Process interview booking if detected, persisting to SQLite3.
        6. Append user & assistant turns to Redis memory.
        7. Return structured ChatResponse.
        """
        conv_id = request.conversation_id or str(uuid.uuid4())

        # Retrieve multi-turn history from Redis
        history: List[ChatMessage] = self.memory_service.get_history(conv_id, limit=12)

        # Vector search in Qdrant for document context
        query_vector = self.embedding_service.embed_query(request.message)
        hits = self.qdrant_service.search_similar(
            query_vector=query_vector,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )

        # Build ChatSource objects for citations
        sources: List[ChatSource] = []
        for hit in hits:
            payload = hit.get("payload", {})
            sources.append(
                ChatSource(
                    chunk_id=str(hit.get("id")),
                    document_id=payload.get("document_id", ""),
                    filename=payload.get("source_filename", "unknown"),
                    text=payload.get("text", ""),
                    score=round(hit.get("score", 0.0), 4),
                    page_number=payload.get("page_number"),
                    chunk_index=payload.get("chunk_index", 0)
                )
            )

        # Call LLM service with context and history
        assistant_text, booking_data, action_taken = await self.llm_service.generate_response(
            user_query=request.message,
            conversation_history=history,
            context_chunks=hits
        )

        booking_record_response: Optional[BookingResponse] = None

        # If an interview was booked, persist to SQLite3 DB
        if booking_data and action_taken == "interview_booking":
            try:
                booking = InterviewBooking(
                    conversation_id=conv_id,
                    name=booking_data["name"],
                    email=booking_data["email"],
                    date=booking_data["date"],
                    time=booking_data["time"],
                    notes=booking_data.get("notes"),
                    status="confirmed"
                )
                db.add(booking)
                db.commit()
                db.refresh(booking)
                booking_record_response = BookingResponse.model_validate(booking)
                logger.info("Persisted interview booking in SQLite3: %s", booking.id)
            except Exception as e:
                db.rollback()
                logger.error("Failed to save booking to SQLite3: %s", e)

        # Persist turns in Redis conversational memory
        self.memory_service.add_message(conv_id, "user", request.message)
        self.memory_service.add_message(
            conv_id,
            "assistant",
            assistant_text,
            metadata={"action": action_taken, "sources_count": len(sources)}
        )

        return ChatResponse(
            conversation_id=conv_id,
            response=assistant_text,
            sources=sources,
            booking=booking_record_response,
            action_taken=action_taken
        )

