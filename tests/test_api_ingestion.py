from unittest.mock import patch
from fastapi.testclient import TestClient


def test_upload_txt_recursive_strategy(client: TestClient):
    file_content = (
        b"Machine learning systems require scalable vector indexing. "
        b"Qdrant provides high-throughput similarity search. "
        b"FastAPI offers blazing fast asynchronous API endpoints."
    )

    with patch("app.services.qdrant_service.QdrantService.upsert_chunks", return_value=True):
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("tech_overview.txt", file_content, "text/plain")},
            data={"chunking_strategy": "recursive_character", "chunk_size": 80, "chunk_overlap": 10}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "tech_overview.txt"
        assert data["file_type"] == "txt"
        assert data["chunking_strategy"] == "recursive_character"
        assert data["chunks_created"] >= 1
        doc_id = data["document_id"]

        # Verify listing
        list_res = client.get("/api/v1/documents/")
        assert list_res.status_code == 200
        docs = list_res.json()
        assert any(d["id"] == doc_id for d in docs)

        # Verify detail
        detail_res = client.get(f"/api/v1/documents/{doc_id}")
        assert detail_res.status_code == 200
        detail_data = detail_res.json()
        assert detail_data["id"] == doc_id
        assert len(detail_data["chunks"]) == data["chunks_created"]


def test_upload_txt_paragraph_strategy(client: TestClient):
    file_content = (
        b"Paragraph one details the ingestion pipeline architecture.\n\n"
        b"Paragraph two outlines vector storage and cosine similarity scoring.\n\n"
        b"Paragraph three describes Redis conversational memory persistence."
    )

    with patch("app.services.qdrant_service.QdrantService.upsert_chunks", return_value=True):
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("architecture.txt", file_content, "text/plain")},
            data={"chunking_strategy": "paragraph_semantic", "chunk_size": 100, "chunk_overlap": 20}
        )

        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "architecture.txt"
        assert data["chunking_strategy"] == "paragraph_semantic"
        assert data["chunks_created"] >= 2


def test_delete_document(client: TestClient):
    file_content = b"Temporary document to be deleted."

    with patch("app.services.qdrant_service.QdrantService.upsert_chunks", return_value=True), \
         patch("app.services.qdrant_service.QdrantService.delete_document_vectors", return_value=True):
        # 1. Upload
        res = client.post(
            "/api/v1/documents/upload",
            files={"file": ("to_delete.txt", file_content, "text/plain")},
            data={"chunking_strategy": "recursive_character"}
        )
        assert res.status_code == 201
        doc_id = res.json()["document_id"]

        # 2. Delete
        del_res = client.delete(f"/api/v1/documents/{doc_id}")
        assert del_res.status_code == 200

        # 3. Verify gone from SQLite
        get_res = client.get(f"/api/v1/documents/{doc_id}")
        assert get_res.status_code == 404

