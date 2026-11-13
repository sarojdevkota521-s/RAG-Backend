import pytest
from app.services.chunker import DocumentChunker
from app.services.extractor import ExtractedPage
from app.schemas.document import ChunkingStrategy


def test_recursive_character_chunking():
    text = (
        "FastAPI is a modern, fast web framework for building APIs with Python 3.8+.\n\n"
        "It provides high performance on par with NodeJS and Go. It has built-in OpenAPI documentation "
        "and automatic data validation via Pydantic.\n\n"
        "Qdrant is an open-source vector similarity search engine. It provides production-ready "
        "service with a convenient API to store, search, and manage vectors along with additional payload."
    )
    pages = [ExtractedPage(text=text, page_number=1)]

    chunks = DocumentChunker.split(
        pages=pages,
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
        chunk_size=120,
        chunk_overlap=20
    )

    assert len(chunks) >= 2
    assert all(c.page_number == 1 for c in chunks)
    assert all(len(c.text) > 0 for c in chunks)
    # Check indices are sequential
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_paragraph_semantic_chunking():
    p1 = "First paragraph discussing the importance of vector databases in enterprise AI pipelines."
    p2 = "Second paragraph covering Redis as a lightning-fast in-memory key-value store for session history."
    p3 = "Third paragraph detailing SQLite3 for zero-configuration, serverless, self-contained SQL database engines."

    text = f"{p1}\n\n{p2}\n\n{p3}"
    pages = [ExtractedPage(text=text, page_number=1)]

    chunks = DocumentChunker.split(
        pages=pages,
        strategy=ChunkingStrategy.PARAGRAPH_SEMANTIC,
        chunk_size=150,
        chunk_overlap=30
    )

    assert len(chunks) >= 2
    assert all(c.page_number == 1 for c in chunks)
    # Check that paragraph contents are preserved without broken words
    assert any("vector databases" in c.text for c in chunks)
    assert any("Redis" in c.text for c in chunks)
    assert any("SQLite3" in c.text for c in chunks)

