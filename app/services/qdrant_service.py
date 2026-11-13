import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_qdrant_client_instance: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Singleton getter for Qdrant client with prefer_grpc=False."""
    global _qdrant_client_instance
    if _qdrant_client_instance is None:
        cleaned_key = settings.clean_key(settings.QDRANT_API_KEY)
        cleaned_url = settings.QDRANT_URL.strip().strip("'\"")

        logger.info("Initializing Qdrant client at %s", cleaned_url)
        _qdrant_client_instance = QdrantClient(
            url=cleaned_url,
            api_key=cleaned_key,
            prefer_grpc=settings.QDRANT_PREFER_GRPC,
            timeout=10.0
        )
    return _qdrant_client_instance


class QdrantService:
    """Service for managing collections, storing embeddings, and performing vector similarity search in Qdrant."""

    def __init__(self, client: Optional[QdrantClient] = None):
        self.client = client or get_qdrant_client()
        self.default_collection = settings.COLLECTION_NAME

    def ensure_collection(self, collection_name: Optional[str] = None) -> bool:
        """Create the target Qdrant collection if it does not already exist."""
        coll = collection_name or self.default_collection
        try:
            exists = self.client.collection_exists(collection_name=coll)
            if not exists:
                logger.info("Creating Qdrant collection '%s' with dim %d", coll, settings.EMBEDDING_DIMENSION)
                self.client.create_collection(
                    collection_name=coll,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
            return True
        except Exception as e:
            logger.error("Failed to ensure collection '%s': %s", coll, e)
            raise

    def upsert_chunks(
        self,
        points: List[PointStruct],
        collection_name: Optional[str] = None
    ) -> bool:
        """Upsert embedded document chunk points into Qdrant."""
        coll = collection_name or self.default_collection
        self.ensure_collection(coll)

        if not points:
            return True

        try:
            self.client.upsert(
                collection_name=coll,
                points=points
            )
            logger.info("Successfully upserted %d points to collection '%s'", len(points), coll)
            return True
        except Exception as e:
            logger.error("Error upserting points to Qdrant collection '%s': %s", coll, e)
            raise

    def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 4,
        score_threshold: float = 0.0,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for top-K most similar chunks in Qdrant given a query vector.
        Returns a list of dictionaries with point id, score, and payload.
        """
        coll = collection_name or self.default_collection
        self.ensure_collection(coll)

        try:
            # In qdrant-client, query_points or search can be used
            # We use query_points if available, falling back to search
            if hasattr(self.client, "query_points"):
                results = self.client.query_points(
                    collection_name=coll,
                    query=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold if score_threshold > 0 else None,
                    with_payload=True
                )
                points = results.points
            else:
                points = self.client.search(
                    collection_name=coll,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold if score_threshold > 0 else None,
                    with_payload=True
                )

            hits = []
            for hit in points:
                hits.append({
                    "id": str(hit.id),
                    "score": float(hit.score),
                    "payload": hit.payload or {}
                })
            return hits
        except Exception as e:
            logger.error("Vector search failed on collection '%s': %s", coll, e)
            return []

    def delete_document_vectors(self, document_id: str, collection_name: Optional[str] = None) -> bool:
        """Delete all chunk vectors associated with a specific document_id."""
        coll = collection_name or self.default_collection
        try:
            if not self.client.collection_exists(coll):
                return True

            self.client.delete(
                collection_name=coll,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            logger.info("Deleted vector points for document '%s' in collection '%s'", document_id, coll)
            return True
        except Exception as e:
            logger.error("Failed to delete points for document '%s': %s", document_id, e)
            return False

    def health_check(self) -> bool:
        """Check Qdrant connectivity."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.warning("Qdrant health check failed: %s", e)
            return False

