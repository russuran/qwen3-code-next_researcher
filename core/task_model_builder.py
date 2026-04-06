"""Task model builder: constructs a TaskModel from a RunSpec via LLM."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from core.llm import LLM, LLMMode

logger = logging.getLogger(__name__)


class TaskModel(BaseModel):
    entities: list[str] = []
    subtasks: list[dict[str, Any]] = []
    dependencies: list[dict[str, str]] = []
    risks: list[str] = []
    artifact_types: list[str] = []
    capabilities: list[str] = []


_PROMPT = """\
Analyze this research/engineering run specification and produce a task model.

Run spec:
  topic: {topic}
  mode: {mode}
  constraints: {constraints}

Return ONLY a JSON object:
{{
  "entities": ["key entities to investigate"],
  "subtasks": [{{"name": "...", "description": "..."}}],
  "dependencies": [{{"from": "subtask_a", "to": "subtask_b"}}],
  "risks": ["potential risk"],
  "artifact_types": ["report", "patch", "benchmark"],
  "capabilities": ["web_search", "code_analysis", "sandbox"]
}}
"""


async def build_task_model(run_spec: dict[str, Any], llm: LLM) -> TaskModel:
    """Build a structured task model from a run specification."""
    prompt = _PROMPT.format(
        topic=run_spec.get("topic", ""),
        mode=run_spec.get("mode", "greenfield"),
        constraints=run_spec.get("constraints", []),
    )
    try:
        result = await llm.generate_structured(prompt, TaskModel, mode=LLMMode.THINKING)
        logger.info(
            "Built task model: %d subtasks, %d capabilities",
            len(result.subtasks), len(result.capabilities),
        )
        return result  # type: ignore[return-value]
    except Exception as e:
        logger.error("Failed to build task model: %s", e)
        return TaskModel(
            entities=[run_spec.get("topic", "unknown")],
            subtasks=[{"name": "research", "description": run_spec.get("topic", "")}],
            capabilities=["web_search"],
        )
