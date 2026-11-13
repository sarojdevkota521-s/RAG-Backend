from fastapi.testclient import TestClient


def test_create_and_get_booking(client: TestClient):
    payload = {
        "name": "Jane Doe",
        "email": "jane.doe@example.com",
        "date": "2026-11-25",
        "time": "15:00",
        "notes": "Backend AI Developer Interview"
    }

    # 1. Create booking
    response = client.post("/api/v1/bookings/", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Jane Doe"
    assert data["email"] == "jane.doe@example.com"
    assert data["date"] == "2026-11-25"
    assert data["time"] == "15:00"
    assert data["status"] == "confirmed"
    booking_id = data["id"]

    # 2. Get booking by ID
    get_res = client.get(f"/api/v1/bookings/{booking_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == booking_id

    # 3. List bookings
    list_res = client.get("/api/v1/bookings/")
    assert list_res.status_code == 200
    bookings = list_res.json()
    assert any(b["id"] == booking_id for b in bookings)

