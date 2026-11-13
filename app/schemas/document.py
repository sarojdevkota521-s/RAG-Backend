from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class ChunkingStrategy(str, Enum):
    """Supported document chunking strategies."""
    RECURSIVE_CHARACTER = "recursive_character"
    PARAGRAPH_SEMANTIC = "paragraph_semantic"


class DocumentChunkResponse(BaseModel):
    """Schema for document chunk details."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    chunk_index: int
    content: str
    vector_id: str
    page_number: Optional[int] = None
    char_count: int
    created_at: datetime


class DocumentResponse(BaseModel):
    """Schema for document metadata list/view."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: str
    file_size: int
    chunking_strategy: str
    chunk_count: int
    created_at: datetime


class DocumentDetailResponse(DocumentResponse):
    """Detailed document response including chunk previews."""
    chunks: List[DocumentChunkResponse] = Field(default_factory=list)


class DocumentIngestResponse(BaseModel):
    """Response returned upon successful document ingestion."""
    message: str
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunking_strategy: str
    chunks_created: int
    qdrant_collection: str
    status: str = "success"

