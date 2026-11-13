import uuid
import logging
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.orm import Session
from qdrant_client.http.models import PointStruct

from app.core.database import get_db
from app.core.config import get_settings
from app.models.document import Document, DocumentChunk
from app.schemas.document import (
    ChunkingStrategy,
    DocumentResponse,
    DocumentDetailResponse,
    DocumentIngestResponse
)
from app.services.extractor import DocumentExtractor
from app.services.chunker import DocumentChunker
from app.services.embedding import EmbeddingService
from app.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/documents", tags=["Document Ingestion"])

embedding_service = EmbeddingService()
qdrant_service = QdrantService()


@router.post(
    "/upload",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a .pdf or .txt document"
)
async def upload_document(
    file: UploadFile = File(..., description="Document file (.pdf or .txt)"),
    chunking_strategy: ChunkingStrategy = Form(
        ChunkingStrategy.RECURSIVE_CHARACTER,
        description="Selectable chunking strategy: 'recursive_character' or 'paragraph_semantic'"
    ),
    chunk_size: int = Form(500, ge=20, le=4000, description="Target character size per chunk"),
    chunk_overlap: int = Form(50, ge=0, le=1000, description="Overlap in characters between consecutive chunks"),
    db: Session = Depends(get_db)
) -> DocumentIngestResponse:
    """
    Ingest a document:
    1. Validates file extension (.pdf or .txt).
    2. Extracts clean text from file bytes.
    3. Applies selected chunking strategy ('recursive_character' or 'paragraph_semantic').
    4. Generates dense embeddings.
    5. Stores vector embeddings in Qdrant.
    6. Stores metadata and chunk references in SQLite3.
    """
    filename = file.filename or "uploaded_file"
    ext = filename.lower().split(".")[-1] if "." in filename else ""

    if ext not in ["pdf", "txt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Only .pdf and .txt are supported."
        )

    content_bytes = await file.read()
    if not content_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    # 1. Extract text
    pages = DocumentExtractor.extract(filename, content_bytes)

    # 2. Chunk text using the selected strategy
    chunks = DocumentChunker.split(
        pages=pages,
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid text chunks could be extracted from the document."
        )

    # 3. Generate embeddings
    chunk_texts = [c.text for c in chunks]
    embeddings = embedding_service.embed_documents(chunk_texts)

    # 4. Prepare Qdrant points
    document_id = str(uuid.uuid4())
    qdrant_points: List[PointStruct] = []
    db_chunks: List[DocumentChunk] = []

    for chunk, vector in zip(chunks, embeddings):
        vector_point_id = str(uuid.uuid4())

        qdrant_points.append(
            PointStruct(
                id=vector_point_id,
                vector=vector,
                payload={
                    "document_id": document_id,
                    "source_filename": filename,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "strategy": chunking_strategy.value
                }
            )
        )

        db_chunks.append(
            DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.text,
                vector_id=vector_point_id,
                page_number=chunk.page_number,
                char_count=chunk.char_count
            )
        )

    # 5. Upsert to Qdrant
    try:
        qdrant_service.upsert_chunks(qdrant_points)
    except Exception as e:
        logger.error("Failed to store vectors in Qdrant: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document vectors in Qdrant: {str(e)}"
        )

    # 6. Save metadata in SQLite3
    try:
        doc_record = Document(
            id=document_id,
            filename=filename,
            file_type=ext,
            file_size=len(content_bytes),
            chunking_strategy=chunking_strategy.value,
            chunk_count=len(db_chunks)
        )
        db.add(doc_record)
        db.flush()

        for chunk_record in db_chunks:
            db.add(chunk_record)

        db.commit()
    except Exception as e:
        db.rollback()
        # Attempt compensation in Qdrant
        qdrant_service.delete_document_vectors(document_id)
        logger.error("Database commit error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist document metadata in SQLite database."
        )

    return DocumentIngestResponse(
        message="Document successfully processed and indexed.",
        document_id=document_id,
        filename=filename,
        file_type=ext,
        file_size_bytes=len(content_bytes),
        chunking_strategy=chunking_strategy.value,
        chunks_created=len(db_chunks),
        qdrant_collection=settings.COLLECTION_NAME,
        status="success"
    )


@router.get(
    "/",
    response_model=List[DocumentResponse],
    summary="List all ingested documents"
)
def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
) -> List[DocumentResponse]:
    """Retrieve metadata of all ingested documents stored in SQLite3."""
    docs = db.query(Document).offset(skip).limit(limit).all()
    return [DocumentResponse.model_validate(doc) for doc in docs]


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details and chunks"
)
def get_document(
    document_id: str,
    db: Session = Depends(get_db)
) -> DocumentDetailResponse:
    """Retrieve full metadata and chunk list for a specific document."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found."
        )
    return DocumentDetailResponse.model_validate(doc)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a document, its chunks, and its vectors"
)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db)
) -> dict:
    """Remove a document from SQLite3 and delete its vectors from Qdrant."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found."
        )

    # Delete from Qdrant
    qdrant_service.delete_document_vectors(document_id)

    # Delete from SQLite3 (cascades to chunks)
    db.delete(doc)
    db.commit()

    return {
        "message": f"Document '{document_id}' and all associated vectors deleted successfully.",
        "document_id": document_id
    }
