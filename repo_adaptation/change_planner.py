"""Change planner: plans code changes before generating diffs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from core.llm import LLM, LLMMode

logger = logging.getLogger(__name__)


class ChangeStep(BaseModel):
    file_path: str
    action: str = "modify"  # modify | create | delete
    description: str = ""
    dependencies: list[str] = []


class ChangePlan(BaseModel):
    task: str = ""
    steps: list[ChangeStep] = []
    affected_files: list[str] = []
    estimated_risk: str = "low"  # low | medium | high


_PLAN_PROMPT = """\
Plan code changes for this task. Task: {task}
Repository files: {file_list}
Return ONLY JSON: {{"steps": [{{"file_path": "...", "action": "modify", "description": "..."}}], \
"affected_files": ["..."], "estimated_risk": "low|medium|high"}}
"""


async def plan_changes(
    task: str,
    repo_path: str | Path,
    llm: LLM,
) -> ChangePlan:
    """Generate a change plan for a given task in a repository."""
    repo = Path(repo_path)
    files = _list_files(repo, max_files=40)

    prompt = _PLAN_PROMPT.format(task=task, file_list="\n".join(f"  {f}" for f in files))

    try:
        result = await llm.generate_structured(prompt, ChangePlan, mode=LLMMode.THINKING)
        result.task = task  # type: ignore[union-attr]
        logger.info(
            "Change plan: %d steps, risk=%s, files=%d",
            len(result.steps),  # type: ignore[union-attr]
            result.estimated_risk,  # type: ignore[union-attr]
            len(result.affected_files),  # type: ignore[union-attr]
        )
        return result  # type: ignore[return-value]
    except Exception as e:
        logger.error("Change planning failed: %s", e)
        return ChangePlan(task=task, estimated_risk="high")


def _list_files(repo: Path, max_files: int = 40) -> list[str]:
    files: list[str] = []
    for f in repo.rglob("*.py"):
        if any(p.startswith(".") or p in ("__pycache__", "venv", ".venv") for p in f.parts):
            continue
        files.append(str(f.relative_to(repo)))
        if len(files) >= max_files:
            break
    return sorted(files)
