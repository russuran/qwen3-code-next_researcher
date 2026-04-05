"""Replay engine: replay executed runs from journal."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ReplayEvent(BaseModel):
    timestamp: str
    phase: str
    action: str
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    result_summary: str = ""


class ReplayRun(BaseModel):
    slug: str
    events: list[ReplayEvent] = []
    phases: dict[str, list[ReplayEvent]] = {}
    total_events: int = 0
    phase_summary: dict[str, int] = {}


class ReplayEngine:
    """Loads and replays runs from JSONL journal files."""

    def __init__(self, journal_dir: str = "journal") -> None:
        self.journal_dir = Path(journal_dir)

    def list_runs(self) -> list[str]:
        if not self.journal_dir.exists():
            return []
        return [f.stem for f in sorted(self.journal_dir.glob("*.jsonl"))]

    def load(self, slug: str) -> ReplayRun:
        path = self.journal_dir / f"{slug}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Journal not found: {path}")

        run = ReplayRun(slug=slug)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    event = ReplayEvent(**{
                        k: v for k, v in data.items()
                        if k in ReplayEvent.model_fields
                    })
                    run.events.append(event)
                    run.phases.setdefault(event.phase, []).append(event)
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning("Failed to parse journal line: %s", e)

        run.total_events = len(run.events)
        run.phase_summary = {phase: len(events) for phase, events in run.phases.items()}
        return run

    def get_phase_events(self, slug: str, phase: str) -> list[ReplayEvent]:
        run = self.load(slug)
        return run.phases.get(phase, [])

    def get_tool_calls(self, slug: str) -> list[ReplayEvent]:
        run = self.load(slug)
        return [e for e in run.events if e.tool_name]

    def get_timeline(self, slug: str) -> list[dict[str, Any]]:
        """Returns a simplified timeline for visualization."""
        run = self.load(slug)
        timeline = []
        for e in run.events:
            timeline.append({
                "time": e.timestamp,
                "phase": e.phase,
                "action": e.action,
                "tool": e.tool_name,
                "summary": e.result_summary[:100],
            })
        return timeline
