import os
os.environ["DISABLE_SQLALCHEMY_CEXT"] = "1"

import pytest
from typing import Generator
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.core.database import Base, get_db

# Use isolated SQLite test database
TEST_DB_URL = "sqlite:///./test_rag_backend.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create test database schema before tests and drop after."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    if os.path.exists("./test_rag_backend.db"):
        try:
            os.remove("./test_rag_backend.db")
        except Exception:
            pass


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Provide a clean database session for each test."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """TestClient fixture with overridden database and mocked external services."""
    from main import app

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    # In-memory mock for Redis storage during unit testing
    mock_redis_storage = {}

    def mock_rpush(key, val):
        mock_redis_storage.setdefault(key, []).append(val)
        return len(mock_redis_storage[key])

    def mock_lrange(key, start, end):
        items = mock_redis_storage.get(key, [])
        if end == -1:
            end = len(items)
        else:
            end = end + 1
        if start < 0:
            start = max(0, len(items) + start)
        return items[start:end]

    def mock_delete(key):
        if key in mock_redis_storage:
            del mock_redis_storage[key]
            return 1
        return 0

    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mock_redis.rpush.side_effect = mock_rpush
    mock_redis.lrange.side_effect = mock_lrange
    mock_redis.delete.side_effect = mock_delete
    mock_redis.expire.return_value = True
    mock_redis.llen.side_effect = lambda k: len(mock_redis_storage.get(k, []))

    with patch("app.services.qdrant_service.QdrantService.ensure_collection", return_value=True), \
         patch("app.services.qdrant_service.QdrantService.health_check", return_value=True), \
         patch("app.core.redis_client.get_redis_client", return_value=mock_redis), \
         patch("app.services.memory_service.get_redis_client", return_value=mock_redis):
        with TestClient(app) as test_client:
            yield test_client

    app.dependency_overrides.clear()
