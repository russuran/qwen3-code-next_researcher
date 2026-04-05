"""Intent interpreter: classifies task type and builds RunSpec."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from core.llm import LLM, LLMMode

logger = logging.getLogger(__name__)


class RunSpec(BaseModel):
    query: str
    mode: str = "greenfield"  # greenfield | brownfield
    task_type: str = "comparative_research"
    # factual_lookup, comparative_research, architecture_design,
    # benchmark_task, repository_adaptation, patch_task
    constraints: list[str] = []
    files: list[str] = []
    urls: list[str] = []
    repository: dict[str, Any] | None = None
    budgets: dict[str, int] = {
        "max_parallel_subtasks": 4,
        "max_parallel_fetches": 8,
        "max_parallel_sandbox_jobs": 2,
    }
    git_policy: dict[str, Any] = {
        "init_if_missing": True,
        "create_branches": True,
        "create_worktrees": False,
        "push_enabled": False,
    }
    allow_code_changes: str = "none"  # none | sandbox_only | repo
    autonomy_level: str = "full"  # full | semi | advisory
    refresh_mode: str = "relaxed"  # strict | relaxed


CLASSIFY_PROMPT = """\
Classify the following user request into a task type and mode.

Request: {query}

Respond with ONLY a JSON object:
{{
  "task_type": "factual_lookup|comparative_research|architecture_design|benchmark_task|repository_adaptation|patch_task",
  "mode": "greenfield|brownfield",
  "constraints": ["constraint if mentioned"],
  "needs_clarification": false,
  "clarification_question": ""
}}

Rules:
- factual_lookup: simple question about a specific fact
- comparative_research: compare methods, tools, approaches
- architecture_design: design a system or solution
- benchmark_task: evaluate/benchmark tools
- repository_adaptation: modify existing code
- patch_task: fix a specific bug or add a feature
- mode=brownfield if a repository/code is mentioned, otherwise greenfield
"""


class IntentInterpreter:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def interpret(self, query: str, repo_url: str | None = None) -> RunSpec:
        prompt = CLASSIFY_PROMPT.format(query=query)
        raw = await self.llm.generate(prompt, mode=LLMMode.FAST)

        # Parse response
        data = self._extract_json(raw)

        task_type = data.get("task_type", "comparative_research")
        mode = data.get("mode", "greenfield")
        constraints = data.get("constraints", [])

        # Override mode if repo is provided
        if repo_url:
            mode = "brownfield"

        spec = RunSpec(
            query=query,
            mode=mode,
            task_type=task_type,
            constraints=constraints,
        )

        if repo_url:
            spec.repository = {"url": repo_url}
            spec.allow_code_changes = "repo"

        logger.info("Interpreted: type=%s mode=%s", task_type, mode)
        return spec

    @staticmethod
    def _extract_json(text: str) -> dict:
        import json
        text = text.strip()
        if "```" in text:
            blocks = text.split("```")
            for block in blocks[1::2]:
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except Exception:
                    continue
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
        return {}
