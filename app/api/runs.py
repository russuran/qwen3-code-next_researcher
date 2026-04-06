from __future__ import annotations

import asyncio
import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import get_db, get_settings
from app.schemas.runs import EventResponse, RunCreate, RunResponse
from app.services import run_service
from app.settings import Settings

router = APIRouter()


@router.post("", response_model=RunResponse, status_code=201)
async def create_run(
    body: RunCreate,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    run = await run_service.create_run(db, body)
    run_service.start_run_background(run.id, body, settings)
    return RunResponse(
        id=str(run.id),
        topic=run.topic,
        status=run.status,
        output_dir=run.output_dir,
        error=run.error,
        created_at=run.created_at,
        updated_at=run.updated_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, db: AsyncSession = Depends(get_db)):
    try:
        uid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID")

    run = await run_service.get_run(db, uid)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return RunResponse(
        id=str(run.id),
        topic=run.topic,
        status=run.status,
        output_dir=run.output_dir,
        error=run.error,
        created_at=run.created_at,
        updated_at=run.updated_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


@router.get("/{run_id}/events", response_model=list[EventResponse])
async def get_events(run_id: str, db: AsyncSession = Depends(get_db)):
    try:
        uid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID")

    run = await run_service.get_run(db, uid)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    events = await run_service.get_run_events(db, uid)
    return [
        EventResponse(
            id=e.id,
            run_id=str(e.run_id),
            phase=e.phase,
            action=e.action,
            tool_name=e.tool_name,
            result_summary=e.result_summary or "",
            created_at=e.created_at,
        )
        for e in events
    ]


@router.get("/{run_id}/events/stream")
async def stream_events(run_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    """SSE endpoint via Redis pub/sub with DB fallback."""
    try:
        uid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID")

    run = await run_service.get_run(db, uid)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_generator():
        from app.services.event_bus import subscribe_events
        # First send existing events from DB
        last_id = 0
        events = await run_service.get_run_events(db, uid)
        for e in events:
            data = json.dumps({
                "id": e.id, "phase": e.phase, "action": e.action,
                "tool_name": e.tool_name, "result_summary": e.result_summary or "",
                "created_at": e.created_at.isoformat() if e.created_at else "",
            })
            yield f"data: {data}\n\n"
            last_id = e.id

        # Then stream live events via Redis pub/sub
        try:
            async for event_data in subscribe_events(run_id):
                if await request.is_disconnected():
                    break
                yield f"data: {json.dumps(event_data)}\n\n"
                # Check if run is done
                current_run = await run_service.get_run(db, uid)
                if current_run and current_run.status in ("completed", "failed", "cancelled"):
                    yield f"data: {json.dumps({'status': current_run.status, 'done': True})}\n\n"
                    break
        except Exception:
            # Fallback to polling if Redis unavailable
            while True:
                if await request.is_disconnected():
                    break
                events = await run_service.get_run_events(db, uid)
                for e in events:
                    if e.id > last_id:
                        data = json.dumps({
                            "id": e.id, "phase": e.phase, "action": e.action,
                            "tool_name": e.tool_name, "result_summary": e.result_summary or "",
                            "created_at": e.created_at.isoformat() if e.created_at else "",
                        })
                        yield f"data: {data}\n\n"
                        last_id = e.id
                current_run = await run_service.get_run(db, uid)
                if current_run and current_run.status in ("completed", "failed", "cancelled"):
                    yield f"data: {json.dumps({'status': current_run.status, 'done': True})}\n\n"
                    break
                await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/{run_id}/cancel")
async def cancel_run(run_id: str, db: AsyncSession = Depends(get_db)):
    try:
        uid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID")

    run = await run_service.get_run(db, uid)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=409, detail=f"Run already {run.status}")

    run.status = "cancelled"
    await db.commit()
    return {"status": "cancelled"}


@router.get("/{run_id}/task-graph")
async def get_task_graph(run_id: str, db: AsyncSession = Depends(get_db)):
    """Return the DAG structure for a run (if available)."""
    try:
        uid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID")

    run = await run_service.get_run(db, uid)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Task graph is stored in run.config if available
    graph_data = (run.config or {}).get("task_graph")
    if not graph_data:
        return {"nodes": [], "edges": []}
    return graph_data


@router.get("/{run_id}/artifacts")
async def get_artifacts(run_id: str, db: AsyncSession = Depends(get_db)):
    try:
        uid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID")

    artifacts = await run_service.get_run_artifacts(db, uid)
    return [
        {
            "id": str(a.id), "kind": a.kind, "filename": a.filename,
            "uri": a.uri, "size_bytes": a.size_bytes,
        }
        for a in artifacts
    ]


@router.post("/{run_id}/start")
async def start_run(run_id: str, db: AsyncSession = Depends(get_db)):
    try:
        uid = UUID(run_id)
    except ValueError:
        raise HTTPException(400, "Invalid run ID")
    run = await run_service.get_run(db, uid)
    if not run:
        raise HTTPException(404, "Run not found")
    if run.status != "pending":
        raise HTTPException(409, f"Run is {run.status}, not pending")
    return {"status": "started", "run_id": run_id}


@router.post("/{run_id}/files")
async def upload_run_files(run_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    try:
        uid = UUID(run_id)
    except ValueError:
        raise HTTPException(400, "Invalid run ID")
    run = await run_service.get_run(db, uid)
    if not run:
        raise HTTPException(404, "Run not found")
    return {"status": "files_accepted", "run_id": run_id}


@router.get("/{run_id}/trace")
async def get_trace(run_id: str, db: AsyncSession = Depends(get_db)):
    try:
        uid = UUID(run_id)
    except ValueError:
        raise HTTPException(400, "Invalid run ID")
    events = await run_service.get_run_events(db, uid)
    return {
        "run_id": run_id,
        "trace": [
            {"id": e.id, "phase": e.phase, "action": e.action,
             "tool_name": e.tool_name, "result_summary": e.result_summary,
             "latency_ms": e.latency_ms, "parent_event_id": e.parent_event_id,
             "created_at": e.created_at.isoformat() if e.created_at else None}
            for e in events
        ],
    }


@router.post("/{run_id}/repositories")
async def add_repository(run_id: str, db: AsyncSession = Depends(get_db)):
    return {"status": "repository_added", "run_id": run_id}


@router.get("/{run_id}/branches")
async def get_branches(run_id: str, db: AsyncSession = Depends(get_db)):
    return {"run_id": run_id, "branches": []}


@router.get("/{run_id}/patches")
async def get_patches(run_id: str, db: AsyncSession = Depends(get_db)):
    return {"run_id": run_id, "patches": []}


@router.post("/{run_id}/patches/{patch_id}/validate")
async def validate_patch(run_id: str, patch_id: str, db: AsyncSession = Depends(get_db)):
    return {"run_id": run_id, "patch_id": patch_id, "status": "validation_queued"}
