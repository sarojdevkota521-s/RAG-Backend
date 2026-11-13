from unittest.mock import patch
from fastapi.testclient import TestClient


def test_conversational_interview_booking_flow(client: TestClient):
    """
    Test multi-turn interview booking where candidate info is collected and persisted.
    Turn 1: User expresses intent with date and time, but missing name & email.
    Turn 2: User provides name and email -> booking is completed!
    """
    with patch("app.services.qdrant_service.QdrantService.search_similar", return_value=[]):
        # Turn 1: Partial booking
        res1 = client.post(
            "/api/v1/chat/",
            json={
                "message": "I would like to book an interview next Monday at 2:00 PM."
            }
        )
        assert res1.status_code == 200
        data1 = res1.json()
        conv_id = data1["conversation_id"]
        assert data1["action_taken"] == "interview_clarification"
        assert "detail" in data1["response"].lower() or "provide" in data1["response"].lower() or "name" in data1["response"].lower()

        # Turn 2: Provide remaining details using the same conversation_id
        res2 = client.post(
            "/api/v1/chat/",
            json={
                "conversation_id": conv_id,
                "message": "My name is John Doe and my email is john.doe@example.com."
            }
        )
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["conversation_id"] == conv_id
        assert data2["action_taken"] == "interview_booking"
        assert data2["booking"] is not None
        assert data2["booking"]["name"] == "John Doe"
        assert data2["booking"]["email"] == "john.doe@example.com"
        assert "2:00 PM" in data2["booking"]["time"]


def test_conversational_history_endpoint(client: TestClient):
    """Test retrieving and clearing chat history from Redis."""
    conv_id = "test-session-12345"

    with patch("app.services.qdrant_service.QdrantService.search_similar", return_value=[]):
        # Send a chat message
        res = client.post(
            "/api/v1/chat/",
            json={
                "conversation_id": conv_id,
                "message": "Hello, how can you help me today?"
            }
        )
        assert res.status_code == 200

        # Retrieve history
        hist_res = client.get(f"/api/v1/chat/history/{conv_id}")
        assert hist_res.status_code == 200
        hist_data = hist_res.json()
        assert hist_data["conversation_id"] == conv_id
        assert hist_data["total_messages"] >= 2  # user + assistant

        # Clear history
        del_res = client.delete(f"/api/v1/chat/history/{conv_id}")
        assert del_res.status_code == 200
        assert del_res.json()["cleared"] is True
