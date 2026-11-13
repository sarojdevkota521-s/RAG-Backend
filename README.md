# Production-Grade Conversational RAG & Ingestion Backend

A clean, modular backend built with **FastAPI**, featuring **Document Ingestion** into **Qdrant** with **SQLite3** metadata persistence, and a **Custom Conversational RAG Pipeline** with **Redis** chat memory and **LLM-driven Interview Booking**.

---

## Architecture Overview

```
                        ┌──────────────────────────────────────────────┐
                        │              FastAPI Application             │
                        │                 (/api/v1)                    │
                        └───────┬──────────────────────────────┬───────┘
                                │                              │
                ┌───────────────▼──────────────┐ ┌─────────────▼───────────────┐
                │    Document Ingestion API    │ │    Conversational RAG API   │
                │  (POST /documents/upload)    │ │        (POST /chat)         │
                └───────┬──────────────┬───────┘ └───────┬─────────────┬───────┘
                        │              │                 │             │
               ┌────────▼────────┐ ┌───▼───────────┐ ┌───▼───────────┐ ┌▼──────────────┐
               │    Chunker      │ │   FastEmbed   │ │     Redis     │ │      LLM      │
               │  (2 Strategies) │ │  (MiniLM-L6)  │ │  Chat Memory  │ │  (Tool Call)  │
               └────────┬────────┘ └───┬───────────┘ └───────────────┘ └───────┬───────┘
                        │              │                                       │
               ┌────────▼────────┐ ┌───▼───────────┐                           │
               │     SQLite3     │ │    Qdrant     │◄──────────────────────────┘
               │  (Metadata DB)  │ │ (Vector Store)│      (Context Search /
               └─────────────────┘ └───────────────┘       Interview Booking)
```

### Key Highlights
- **Document Ingestion API**:
  - Supports `.pdf` (via PyMuPDF/pypdf) and `.txt` files.
  - **Two Selectable Chunking Strategies**:
    1. `recursive_character`: Hierarchical recursive character splitting with overlap.
    2. `paragraph_semantic`: Semantic paragraph-based chunking with sliding context windowing.
  - Generates dense vector embeddings (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions) via ONNX-backed FastEmbed.
  - Stores vectors in **Qdrant** and document/chunk metadata in **SQLite3**.
- **Conversational RAG API**:
  - **Custom RAG (No `RetrievalQAChain`)**: Transparent, controllable orchestration.
  - **Redis Chat Memory**: Preserves multi-turn conversation context keyed by `conversation_id` with configurable TTL (`CHAT_MEMORY_TTL`).
  - **LLM Tool Calling & Extraction**: Natively detects candidate interview booking intent and extracts `name`, `email`, `date`, and `time`. Prompts for missing fields across multi-turn interactions.
  - **Persistent Bookings**: Stores confirmed bookings in SQLite3 with audit timestamps and status.
  - **Dual Engine Execution**: Compatible with OpenAI, OpenRouter, Groq, and xAI with a resilient offline fallback engine.

---

## Directory Structure

```
Rag-Backend/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── booking.py          # Interview booking CRUD routes
│   │       ├── chat.py             # Conversational RAG & history routes
│   │       ├── ingestion.py        # File upload & ingestion routes
│   │       └── router.py           # V1 route aggregator
│   ├── core/
│   │   ├── config.py               # Pydantic Settings & environment config
│   │   ├── database.py             # SQLite3 engine, Base, and session dependency
│   │   └── redis_client.py         # Redis connection management
│   ├── models/
│   │   ├── booking.py              # SQLAlchemy InterviewBooking model
│   │   └── document.py             # SQLAlchemy Document & DocumentChunk models
│   ├── schemas/
│   │   ├── booking.py              # Pydantic models for bookings
│   │   ├── chat.py                 # Pydantic models for chat requests/responses
│   │   └── document.py             # Pydantic models for document ingestion
│   └── services/
│       ├── chunker.py              # 2 selectable chunking strategies
│       ├── embedding.py            # Vector embedding generator
│       ├── extractor.py            # PDF and TXT text extraction
│       ├── llm_service.py          # LLM completions & tool calling
│       ├── memory_service.py       # Redis conversation memory service
│       ├── qdrant_service.py       # Qdrant client, collections & search
│       └── rag_service.py          # Custom RAG orchestration
├── tests/
│   ├── conftest.py                 # Pytest fixtures & isolated test DB
│   ├── test_api_booking.py         # Booking endpoint tests
│   ├── test_api_chat.py            # Multi-turn chat & booking tests
│   ├── test_api_ingestion.py       # Document ingestion tests
│   ├── test_chunker.py             # Chunking strategies unit tests
│   └── test_extractor.py           # Text extraction unit tests
├── .env.example                    # Template environment variables
├── .env                            # Active environment configuration
├── main.py                         # FastAPI entrypoint, lifespan, CORS & health
└── requirements.txt                # Project dependencies
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Redis server running locally (or remote Redis URL)
- Qdrant Cloud or local Qdrant instance

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration (`.env`)
Create a `.env` file based on `.env.example`:
```ini
APP_NAME=Conversational RAG API
DEBUG=True

# SQLite3 Database
DATABASE_URL=sqlite:///./rag_backend.db

# Redis Memory Cache
REDIS_URL=redis://localhost:6379/0
REDIS_PROTOCOL=2
CHAT_MEMORY_TTL=86400

# Qdrant Vector Store
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
COLLECTION_NAME=documents

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# LLM Configuration (OpenAI, OpenRouter, Groq, xAI)
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=meta-llama/llama-3.1-8b-instruct
```

### 4. Run Application
```bash
python main.py
# or using uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## API Reference & Usage Examples

### 1. Document Ingestion API

#### `POST /api/v1/documents/upload`
Uploads `.pdf` or `.txt` files, extracts content, applies chunking, generates embeddings, saves vectors in Qdrant, and stores metadata in SQLite3.

**Parameters (Form Data)**:
- `file`: File upload (`.pdf` or `.txt`)
- `chunking_strategy`: `recursive_character` or `paragraph_semantic` (default: `recursive_character`)
- `chunk_size`: Target character size per chunk (default: `500`)
- `chunk_overlap`: Character overlap between chunks (default: `50`)

**Example cURL**:
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@sample.pdf" \
  -F "chunking_strategy=paragraph_semantic" \
  -F "chunk_size=400" \
  -F "chunk_overlap=40"
```

**Response**:
```json
{
  "message": "Document successfully processed and indexed.",
  "document_id": "b9e1ece9-6aaa-414b-b496-e51bb0d2025a",
  "filename": "sample.pdf",
  "file_type": "pdf",
  "file_size_bytes": 104523,
  "chunking_strategy": "paragraph_semantic",
  "chunks_created": 12,
  "qdrant_collection": "documents",
  "status": "success"
}
```

#### `GET /api/v1/documents/`
List all ingested documents.

#### `GET /api/v1/documents/{document_id}`
Retrieve document metadata and all chunk references.

#### `DELETE /api/v1/documents/{document_id}`
Delete a document and its chunks from SQLite3 and removes its vectors from Qdrant.

---

### 2. Conversational RAG API

#### `POST /api/v1/chat/`
Multi-turn conversational RAG query and interview booking.

**Request Body**:
```json
{
  "message": "What dual chunking strategies are supported?",
  "conversation_id": "optional-uuid",
  "top_k": 4,
  "score_threshold": 0.0
}
```

**Response (Document RAG)**:
```json
{
  "conversation_id": "4ec5eb49-5f1d-4467-8898-d8bbce1587bb",
  "response": "Based on the uploaded technical specification, the system supports dual chunking strategies: recursive character splitting for general prose, and semantic paragraph windowing for structured content.",
  "sources": [
    {
      "chunk_id": "point-uuid-1",
      "document_id": "doc-uuid-1",
      "filename": "specification.txt",
      "text": "Antigravity supports dual chunking strategies...",
      "score": 0.8845,
      "page_number": 1,
      "chunk_index": 0
    }
  ],
  "booking": null,
  "action_taken": "rag_query"
}
```

#### Multi-Turn Interview Booking Flow
**Turn 1: Candidate expresses intent with partial info**
```bash
curl -X POST "http://localhost:8000/api/v1/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I would like to schedule an interview next Tuesday at 2:00 PM."
  }'
```
*Response*:
```json
{
  "conversation_id": "session-123",
  "response": "I would be glad to schedule your interview! To proceed, please provide the following missing detail(s): full name, email address.",
  "sources": [],
  "booking": null,
  "action_taken": "interview_clarification"
}
```

**Turn 2: Candidate provides remaining details**
```bash
curl -X POST "http://localhost:8000/api/v1/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "session-123",
    "message": "My name is Alice Smith and my email is alice.smith@example.com."
  }'
```
*Response*:
```json
{
  "conversation_id": "session-123",
  "response": "I have successfully scheduled your interview for next Tuesday at 2:00 PM. A confirmation email will be sent to alice.smith@example.com.",
  "sources": [],
  "booking": {
    "id": "b6f68748-0c32-42fe-bd18-912fbbd279ae",
    "conversation_id": "session-123",
    "name": "Alice Smith",
    "email": "alice.smith@example.com",
    "date": "next Tuesday",
    "time": "2:00 PM",
    "notes": null,
    "status": "confirmed",
    "created_at": "2026-11-13T21:00:00Z"
  },
  "action_taken": "interview_booking"
}
```

#### `GET /api/v1/chat/history/{conversation_id}`
Retrieve all messages in the conversation session from Redis.

#### `DELETE /api/v1/chat/history/{conversation_id}`
Purge the conversation history from Redis.

---

### 3. Interview Booking API

- `POST /api/v1/bookings/`: Direct manual booking registration.
- `GET /api/v1/bookings/`: List all booked candidate interviews from SQLite3.
- `GET /api/v1/bookings/{booking_id}`: Retrieve details for a specific booking.

---

### 4. Health Check

#### `GET /health`
```json
{
  "status": "healthy",
  "components": {
    "sqlite": "connected",
    "redis": "connected",
    "qdrant": "connected"
  }
}
```

---

## Testing

Execute the complete automated test suite with pytest:
```bash
python -m pytest -v
```

All 11 unit & integration tests verify:
- Chunking strategies (Recursive & Paragraph Semantic)
- Document text extraction from text and PDF
- Upload endpoints with chunking and SQLite metadata persistence
- Multi-turn Redis chat memory
- LLM interview booking intent and slot extraction
- Direct booking CRUD operations

