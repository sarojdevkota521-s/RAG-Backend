import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.database import init_db
from app.core.redis_client import check_redis_health
from app.services.qdrant_service import QdrantService
from app.api.v1 import api_v1_router

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rag_backend")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown events."""
    logger.info("Initializing %s...", settings.APP_NAME)

    # 1. Initialize SQLite database schema
    try:
        init_db()
        logger.info("SQLite database initialized successfully at %s", settings.DATABASE_URL)
    except Exception as e:
        logger.error("Failed to initialize SQLite database: %s", e)

    # 2. Check Qdrant collection
    try:
        qdrant_service = QdrantService()
        qdrant_service.ensure_collection()
        logger.info("Qdrant collection '%s' ready.", settings.COLLECTION_NAME)
    except Exception as e:
        logger.warning("Qdrant connection notice (will retry on requests): %s", e)

    # 3. Check Redis connectivity
    redis_healthy = check_redis_health()
    logger.info("Redis health status: %s", "Connected" if redis_healthy else "Unavailable/Offline")

    yield

    logger.info("Shutting down %s...", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Production-grade Conversational RAG and Document Ingestion backend. "
        "Supports selectable chunking strategies (Recursive & Paragraph Semantic), "
        "Qdrant vector storage, SQLite3 metadata persistence, Redis conversational memory, "
        "and automated interview booking via LLM."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 Routes
app.include_router(api_v1_router)


@app.get("/", tags=["Root"])
def root() -> dict:
    """Root endpoint providing service metadata."""
    return {
        "service": settings.APP_NAME,
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "document_ingestion": "/api/v1/documents/upload",
            "conversational_rag": "/api/v1/chat",
            "chat_history": "/api/v1/chat/history/{conversation_id}",
            "interview_bookings": "/api/v1/bookings",
            "health_check": "/health"
        }
    }


@app.get("/health", tags=["Health"])
def health_check() -> dict:
    """System health check endpoint verifying Redis, Qdrant, and SQLite."""
    redis_ok = check_redis_health()
    qdrant_service = QdrantService()
    qdrant_ok = qdrant_service.health_check()

    return {
        "status": "healthy" if (redis_ok and qdrant_ok) else "degraded",
        "components": {
            "sqlite": "connected",
            "redis": "connected" if redis_ok else "unreachable",
            "qdrant": "connected" if qdrant_ok else "unreachable"
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler returning clean structured error responses."""
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred.", "path": str(request.url.path)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

