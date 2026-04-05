from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class RunCreate(BaseModel):
    topic: str
    mode: str = "greenfield"  # greenfield | brownfield
    sources: list[str] | None = None
    max_results_per_source: int | None = None
    verbose: bool = False
    autonomy_level: str = "full"  # full | semi | advisory
    repository_url: str | None = None
    target_files: list[str] | None = None
    constraints: list[str] | None = None
    allow_code_changes: str = "none"  # none | sandbox_only | repo
    refresh_mode: str = "relaxed"  # strict | relaxed


class RunResponse(BaseModel):
    id: str
    topic: str
    status: str
    output_dir: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    model_config = {"from_attributes": True}


class EventResponse(BaseModel):
    id: int
    run_id: str
    phase: str
    action: str
    tool_name: str | None = None
    result_summary: str = ""
    created_at: datetime

    model_config = {"from_attributes": True}
