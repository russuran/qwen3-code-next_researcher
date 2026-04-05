from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Artifact, Event, Run
from app.db import session as db_session
from app.deps import build_llm
from app.schemas.runs import RunCreate
from app.settings import Settings
from core.agent import AgentConfig, JournalEntry, ResearchAgent

logger = logging.getLogger(__name__)


class InstrumentedAgent(ResearchAgent):
    """Subclass that writes journal entries to both JSONL and Postgres."""

    def __init__(self, *args, run_id: UUID, **kwargs):
        super().__init__(*args, **kwargs)
        self._run_id = run_id

    def _log(self, entry: JournalEntry) -> None:
        super()._log(entry)
        # Fire-and-forget DB write + Redis pub/sub
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._persist_event(entry))
            loop.create_task(self._publish_event(entry))
        except RuntimeError:
            pass

    async def _publish_event(self, entry: JournalEntry) -> None:
        from app.services.event_bus import publish_event
        await publish_event(str(self._run_id), {
            "phase": entry.phase, "action": entry.action,
            "tool_name": entry.tool_name, "result_summary": entry.result_summary,
        })

    async def _persist_event(self, entry: JournalEntry) -> None:
        if db_session.async_session_factory is None:
            return
        try:
            async with db_session.async_session_factory() as db:
                event = Event(
                    run_id=self._run_id,
                    phase=entry.phase,
                    action=entry.action,
                    tool_name=entry.tool_name,
                    tool_input=entry.tool_input,
                    result_summary=entry.result_summary,
                )
                db.add(event)
                await db.commit()
        except Exception as e:
            logger.warning("Failed to persist event: %s", e)


async def create_run(db: AsyncSession, body: RunCreate) -> Run:
    run = Run(topic=body.topic, status="pending", config=body.model_dump())
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return run


async def get_run(db: AsyncSession, run_id: UUID) -> Run | None:
    return await db.get(Run, run_id)


async def get_run_events(db: AsyncSession, run_id: UUID) -> list[Event]:
    result = await db.execute(
        select(Event).where(Event.run_id == run_id).order_by(Event.created_at)
    )
    return list(result.scalars().all())


async def get_run_artifacts(db: AsyncSession, run_id: UUID) -> list[Artifact]:
    result = await db.execute(
        select(Artifact).where(Artifact.run_id == run_id).order_by(Artifact.created_at)
    )
    return list(result.scalars().all())


def start_run_background(run_id: UUID, body: RunCreate, settings: Settings) -> None:
    """Launch agent execution as a background asyncio task."""
    asyncio.create_task(_execute_run(run_id, body, settings))


async def _execute_run(run_id: UUID, body: RunCreate, settings: Settings) -> None:
    if db_session.async_session_factory is None:
        logger.error("DB not initialized, cannot execute run %s", run_id)
        return

    async with db_session.async_session_factory() as db:
        run = await db.get(Run, run_id)
        if not run:
            logger.error("Run %s not found", run_id)
            return

        try:
            run.status = "planning"
            run.started_at = datetime.now(timezone.utc)
            run.updated_at = datetime.now(timezone.utc)
            await db.commit()

            llm = build_llm(settings)

            agent_cfg = AgentConfig(
                sources=body.sources or settings.default_sources,
                max_results_per_source=body.max_results_per_source or settings.max_results_per_source,
                verbose=body.verbose,
                parallel_search=True,
            )

            agent = InstrumentedAgent(
                config=agent_cfg,
                llm=llm,
                run_id=run_id,
            )

            output_path = await agent.run(body.topic)

            run.status = "completed"
            run.output_dir = str(output_path)
            run.finished_at = datetime.now(timezone.utc)

        except Exception as e:
            logger.error("Run %s failed: %s", run_id, e, exc_info=True)
            run.status = "failed"
            run.error = str(e)
            run.finished_at = datetime.now(timezone.utc)

        run.updated_at = datetime.now(timezone.utc)
        await db.commit()
