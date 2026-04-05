"""Brownfield API: repository adaptation endpoints."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Run
from app.deps import get_db, get_settings
from app.services import brownfield_service
from app.settings import Settings

router = APIRouter()


class BrownfieldRequest(BaseModel):
    repo_path: str
    change_request: str
    target_files: list[str]


class BrownfieldResponse(BaseModel):
    id: str
    status: str
    topic: str


@router.post("", response_model=BrownfieldResponse, status_code=201)
async def create_brownfield_run(
    body: BrownfieldRequest,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    run = Run(
        topic=f"brownfield: {body.change_request[:80]}",
        status="pending",
        config={
            "mode": "brownfield",
            "repo_path": body.repo_path,
            "change_request": body.change_request,
            "target_files": body.target_files,
        },
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    asyncio.create_task(
        brownfield_service.execute_brownfield(
            run.id, body.repo_path, body.change_request,
            body.target_files, settings,
        )
    )

    return BrownfieldResponse(id=str(run.id), status=run.status, topic=run.topic)
