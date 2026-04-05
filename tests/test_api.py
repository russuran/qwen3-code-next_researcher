from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    """Create FastAPI app with mocked infrastructure."""
    with patch("app.main.init_engine"), \
         patch("app.main.aioredis") as mock_redis_mod:
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_redis_mod.from_url.return_value = mock_redis

        from app.main import create_app
        application = create_app()
        yield application


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.get = AsyncMock(return_value=None)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def client(app, mock_session):
    """TestClient with mocked DB session."""
    from app.deps import get_db

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_openapi_schema(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = list(resp.json()["paths"].keys())
    assert "/health" in paths
    assert "/runs" in paths


def test_get_run_invalid_id(client):
    resp = client.get("/runs/not-a-uuid")
    assert resp.status_code == 400


def test_get_run_not_found(client):
    resp = client.get("/runs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_create_run(client, mock_session):
    run_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    mock_run = MagicMock()
    mock_run.id = run_id
    mock_run.topic = "test topic"
    mock_run.status = "pending"
    mock_run.config = {}
    mock_run.output_dir = None
    mock_run.error = None
    mock_run.created_at = now
    mock_run.updated_at = now
    mock_run.started_at = None
    mock_run.finished_at = None

    with patch("app.api.runs.run_service") as mock_svc:
        mock_svc.create_run = AsyncMock(return_value=mock_run)
        mock_svc.start_run_background = MagicMock()

        resp = client.post("/runs", json={"topic": "test topic"})

    assert resp.status_code == 201
    data = resp.json()
    assert data["topic"] == "test topic"
    assert data["status"] == "pending"
    assert data["id"] == str(run_id)


def test_get_run_found(client, mock_session):
    run_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    mock_run = MagicMock()
    mock_run.id = run_id
    mock_run.topic = "found topic"
    mock_run.status = "completed"
    mock_run.config = {}
    mock_run.output_dir = "/output/test"
    mock_run.error = None
    mock_run.created_at = now
    mock_run.updated_at = now
    mock_run.started_at = now
    mock_run.finished_at = now

    mock_session.get = AsyncMock(return_value=mock_run)

    with patch("app.api.runs.run_service") as mock_svc:
        mock_svc.get_run = AsyncMock(return_value=mock_run)

        resp = client.get(f"/runs/{run_id}")

    assert resp.status_code == 200
    data = resp.json()
    assert data["topic"] == "found topic"
    assert data["status"] == "completed"
