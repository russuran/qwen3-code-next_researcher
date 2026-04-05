"""Overnight API: autonomous research-to-implementation pipeline."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from uuid import UUID

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Run, Event
from app.db import session as db_session
from app.deps import get_db, get_settings
from app.settings import Settings

router = APIRouter()


class OvernightRequest(BaseModel):
    topic: str
    libraries: str = "pytesseract, easyocr, Pillow, opencv-python"
    num_samples: int = 50
    max_iterations: int = 3
    sources: list[str] = ["github", "arxiv"]


@router.post("", status_code=201)
async def start_overnight(
    body: OvernightRequest,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    run = Run(
        topic=f"overnight: {body.topic[:80]}",
        status="pending",
        mode="greenfield",
        config=body.model_dump(),
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    task = asyncio.create_task(_run_overnight(run.id, body, settings))
    task.add_done_callback(lambda t: logger.error("Overnight task error: %s", t.exception()) if t.exception() else None)

    return {
        "run_id": str(run.id),
        "topic": body.topic,
        "status": "started",
        "message": "Overnight pipeline launched. Check /runs/{id} for progress.",
    }


async def _run_overnight(run_id: UUID, body: OvernightRequest, settings: Settings):
    from core.overnight_pipeline import OvernightPipeline
    from app.deps import build_llm

    if db_session.async_session_factory is None:
        return

    async with db_session.async_session_factory() as db:
        run = await db.get(Run, run_id)
        if not run:
            return

        try:
            run.status = "planning"
            run.started_at = datetime.now(timezone.utc)
            await db.commit()

            llm = build_llm(settings)
            pipeline = OvernightPipeline(
                llm=llm,
                workspace=f"workspace/overnight-{str(run_id)[:8]}",
                max_improvement_iterations=body.max_iterations,
            )

            async def on_progress(phase, message):
                event = Event(
                    run_id=run_id,
                    phase=phase[:20],
                    action="overnight_progress",
                    result_summary=message[:200],
                )
                db.add(event)
                await db.commit()

            results = await pipeline.run(
                topic=body.topic,
                libraries=body.libraries,
                num_samples=body.num_samples,
                sources=body.sources,
                on_progress=on_progress,
            )

            run.status = "completed"
            run.output_dir = results.get("report_path", "").replace("/final_report.md", "")
            run.finished_at = datetime.now(timezone.utc)

            # Final event
            best_acc = results.get("final_metrics", {}).get("accuracy", "N/A")
            event = Event(
                run_id=run_id,
                phase="synthesize",
                action="overnight_complete",
                result_summary=f"Best accuracy: {best_acc}. Report: {results.get('report_path', '')}",
            )
            db.add(event)

        except Exception as e:
            run.status = "failed"
            run.error = str(e)[:500]
            run.finished_at = datetime.now(timezone.utc)

        run.updated_at = datetime.now(timezone.utc)
        await db.commit()
