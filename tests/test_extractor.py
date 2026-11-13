import pytest
from fastapi import HTTPException
from app.services.extractor import DocumentExtractor


def test_extract_txt_utf8():
    content = "Hello, this is a sample test document for ingestion.".encode("utf-8")
    pages = DocumentExtractor.extract("test.txt", content)
    assert len(pages) == 1
    assert "sample test document" in pages[0].text
    assert pages[0].page_number == 1


def test_extract_txt_empty_fails():
    with pytest.raises(HTTPException) as exc_info:
        DocumentExtractor.extract("empty.txt", b"   ")
    assert exc_info.value.status_code == 400


def test_extract_unsupported_extension():
    with pytest.raises(HTTPException) as exc_info:
        DocumentExtractor.extract("image.png", b"fake image bytes")
    assert exc_info.value.status_code == 400

