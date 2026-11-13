import logging
from typing import List, Optional
from fastembed import TextEmbedding
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_embedding_instance: Optional[TextEmbedding] = None


def get_embedding_model() -> TextEmbedding:
    """Singleton getter for the ONNX-backed FastEmbed embedding model."""
    global _embedding_instance
    if _embedding_instance is None:
        logger.info("Initializing embedding model: %s", settings.EMBEDDING_MODEL)
        _embedding_instance = TextEmbedding(model_name=settings.EMBEDDING_MODEL)
    return _embedding_instance


class EmbeddingService:
    """Service to generate dense vector embeddings for documents and search queries."""

    def __init__(self, model: Optional[TextEmbedding] = None):
        self.model = model or get_embedding_model()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of document chunks."""
        if not texts:
            return []
        embeddings_generator = self.model.embed(texts)
        # Convert numpy arrays to Python standard list of floats for serialization & Qdrant
        return [embedding.tolist() for embedding in embeddings_generator]

    def embed_query(self, text: str) -> List[float]:
        """Generate a single vector embedding for a user query."""
        if not text or not text.strip():
            return [0.0] * settings.EMBEDDING_DIMENSION
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else [0.0] * settings.EMBEDDING_DIMENSION

