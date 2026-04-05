"""Replay API: replay executed runs from journal."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from core.replay import ReplayEngine

router = APIRouter()
_engine = ReplayEngine()


@router.get("")
async def list_replays():
    return {"runs": _engine.list_runs()}


@router.get("/{slug}")
async def get_replay(slug: str):
    try:
        run = _engine.load(slug)
        return {
            "slug": run.slug,
            "total_events": run.total_events,
            "phase_summary": run.phase_summary,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Journal not found")


@router.get("/{slug}/timeline")
async def get_timeline(slug: str):
    try:
        return {"timeline": _engine.get_timeline(slug)}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Journal not found")


@router.get("/{slug}/phase/{phase}")
async def get_phase(slug: str, phase: str):
    try:
        events = _engine.get_phase_events(slug, phase)
        return {"phase": phase, "events": [e.model_dump() for e in events]}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Journal not found")


@router.get("/{slug}/tools")
async def get_tool_calls(slug: str):
    try:
        tools = _engine.get_tool_calls(slug)
        return {"tool_calls": [t.model_dump() for t in tools]}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Journal not found")
