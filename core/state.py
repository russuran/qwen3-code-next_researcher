"""Run state management with persistence and recovery."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RunState(BaseModel):
    run_id: str
    status: str = "pending"  # pending | running | completed | failed | paused
    current_phase: str = ""
    completed_nodes: list[str] = []
    failed_nodes: list[str] = []
    metadata: dict[str, str] = {}


def save_state(state: RunState, state_dir: str = "journal") -> Path:
    """Persist run state to disk as JSON."""
    path = Path(state_dir) / f"{state.run_id}.state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    logger.debug("Saved state for run %s -> %s", state.run_id, path)
    return path


def load_state(run_id: str, state_dir: str = "journal") -> RunState | None:
    """Load persisted run state from disk."""
    path = Path(state_dir) / f"{run_id}.state.json"
    if not path.exists():
        logger.debug("No state file for run %s", run_id)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        state = RunState.model_validate(data)
        logger.info("Loaded state for run %s: status=%s", run_id, state.status)
        return state
    except Exception as e:
        logger.error("Failed to load state for %s: %s", run_id, e)
        return None


def recover_state(run_id: str, state_dir: str = "journal") -> RunState:
    """Load state and prepare it for resumption, resetting running nodes."""
    state = load_state(run_id, state_dir)
    if state is None:
        logger.warning("No state to recover for %s, creating fresh", run_id)
        return RunState(run_id=run_id)

    if state.status == "running":
        state.status = "paused"
        logger.info("Recovered run %s: reset from running to paused", run_id)

    return state
