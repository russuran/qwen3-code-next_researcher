from __future__ import annotations

import uuid

from app.db.models import Artifact, Base, Event, Run


def test_run_model_fields():
    run = Run(id=uuid.uuid4(), topic="test topic", status="pending")
    assert run.topic == "test topic"
    assert run.status == "pending"
    assert run.output_dir is None
    assert run.error is None


def test_event_model_fields():
    run_id = uuid.uuid4()
    event = Event(run_id=run_id, phase="search", action="tool_call", tool_name="search_arxiv")
    assert event.phase == "search"
    assert event.tool_name == "search_arxiv"


def test_artifact_model_fields():
    run_id = uuid.uuid4()
    artifact = Artifact(
        run_id=run_id, kind="synthesis", filename="07_synthesis.md",
        uri="file:///app/output/test/07_synthesis.md",
    )
    assert artifact.kind == "synthesis"
    assert "synthesis" in artifact.uri


def test_base_metadata_has_tables():
    table_names = set(Base.metadata.tables.keys())
    assert "runs" in table_names
    assert "events" in table_names
    assert "artifacts" in table_names
