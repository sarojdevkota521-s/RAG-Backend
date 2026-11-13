from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from app.schemas.booking import BookingResponse


class ChatMessage(BaseModel):
    """A single message in conversational history."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Request payload for conversational RAG."""
    message: str = Field(..., min_length=1, max_length=4000, description="User prompt or query")
    conversation_id: Optional[str] = Field(None, description="Optional UUID to continue previous multi-turn conversation")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of most relevant chunks to retrieve from Qdrant")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum cosine similarity score")


class ChatSource(BaseModel):
    """Source citation from retrieved document chunk."""
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float
    page_number: Optional[int] = None
    chunk_index: int


class ChatResponse(BaseModel):
    """Response payload returned by the conversational RAG endpoint."""
    conversation_id: str
    response: str
    sources: List[ChatSource] = Field(default_factory=list)
    booking: Optional[BookingResponse] = None
    action_taken: str = Field(
        ...,
        description="Type of action performed: 'rag_query', 'interview_booking', 'interview_clarification', or 'general_chat'"
    )


class ChatHistoryResponse(BaseModel):
    """Response payload for conversation history."""
    conversation_id: str
    messages: List[ChatMessage]
    total_messages: int

