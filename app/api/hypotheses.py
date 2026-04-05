"""Hypotheses API: execute implementation loop from research results."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Run, Event
from app.db import session as db_session
from app.deps import get_db, get_settings
from app.settings import Settings
from core.hypothesis_loop import HypothesisRegistry, TrackedHypothesis

router = APIRouter()


@router.get("/{run_id}")
async def get_hypotheses(run_id: str, db: AsyncSession = Depends(get_db)):
    """Get hypotheses from a completed research run."""
    try:
        uid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID")

    run = await db.get(Run, uid)
    if not run or not run.output_dir:
        raise HTTPException(status_code=404, detail="Run not found or no output")

    hyp_path = Path(run.output_dir) / "04_hypotheses.json"
    if not hyp_path.exists():
        return {"hypotheses": [], "gaps": [], "message": "No hypotheses generated"}

    data = json.loads(hyp_path.read_text(encoding="utf-8"))
    return data


@router.post("/{run_id}/execute")
async def execute_hypotheses(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Execute implementation loop for hypotheses from a research run."""
    try:
        uid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID")

    run = await db.get(Run, uid)
    if not run or not run.output_dir:
        raise HTTPException(status_code=404, detail="Run not found or no output")

    hyp_path = Path(run.output_dir) / "04_hypotheses.json"
    if not hyp_path.exists():
        raise HTTPException(status_code=404, detail="No hypotheses found")

    data = json.loads(hyp_path.read_text(encoding="utf-8"))
    raw_hypotheses = data.get("hypotheses", [])
    if not raw_hypotheses:
        raise HTTPException(status_code=404, detail="No hypotheses to execute")

    # Build registry
    registry = HypothesisRegistry()
    for h in raw_hypotheses:
        registry.add(TrackedHypothesis(
            id=h.get("id", "H0"),
            title=h.get("title", ""),
            description=h.get("description", ""),
            approach=h.get("approach", ""),
            expected_outcome=h.get("expected_outcome", ""),
            validation_method=h.get("validation_method", ""),
            priority=h.get("priority", 0),
            effort=h.get("effort", "medium"),
            based_on=h.get("based_on", []),
        ))

    # Create implementation run
    impl_run = Run(
        topic=f"implement hypotheses from: {run.topic[:60]}",
        status="pending",
        mode="brownfield",
        config={"source_run_id": str(run_id), "hypotheses_count": len(raw_hypotheses)},
    )
    db.add(impl_run)
    await db.commit()
    await db.refresh(impl_run)

    # Launch in background
    asyncio.create_task(_run_implementation(impl_run.id, registry, settings))

    return {
        "implementation_run_id": str(impl_run.id),
        "hypotheses_count": len(raw_hypotheses),
        "status": "started",
    }


async def _run_implementation(run_id: UUID, registry: HypothesisRegistry, settings: Settings):
    """Background task: run implementation loop."""
    from core.implementation_loop import ImplementationLoop
    from app.deps import build_llm
    from datetime import datetime, timezone

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
            loop = ImplementationLoop(llm=llm, workspace="workspace")

            async def on_progress(hyp, result):
                event = Event(
                    run_id=run_id,
                    phase="analyze",
                    action=f"hypothesis_{result.status}",
                    result_summary=f"H{hyp.id}: {hyp.title[:50]} -> {result.status}",
                )
                db.add(event)
                await db.commit()

            results = await loop.run(registry, on_progress=on_progress)

            # Save results
            output_dir = Path(f"output/implementation-{str(run_id)[:8]}")
            output_dir.mkdir(parents=True, exist_ok=True)

            (output_dir / "implementation_results.json").write_text(
                json.dumps([r.model_dump() for r in results], indent=2, default=str),
                encoding="utf-8",
            )
            registry.save(output_dir / "hypothesis_registry.json")

            # Summary
            validated = sum(1 for r in results if r.status == "validated")
            rejected = sum(1 for r in results if r.status == "rejected")

            event = Event(
                run_id=run_id,
                phase="synthesize",
                action="implementation_complete",
                result_summary=f"{validated} validated, {rejected} rejected out of {len(results)}",
            )
            db.add(event)

            run.status = "completed"
            run.output_dir = str(output_dir)
            run.finished_at = datetime.now(timezone.utc)

        except Exception as e:
            run.status = "failed"
            run.error = str(e)
            run.finished_at = datetime.now(timezone.utc)

        run.updated_at = datetime.now(timezone.utc)
        await db.commit()
