"""Meta agent: improvement loop for prompts, policies, and thresholds."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel

from core.llm import LLM, LLMMode

logger = logging.getLogger(__name__)


class Improvement(BaseModel):
    improvement_id: str = ""
    target: str  # prompt, policy, threshold, config
    description: str
    before: str = ""
    after: str = ""
    validated: bool = False
    metric_delta: float = 0.0
    created_at: str = ""


IMPROVEMENT_PROMPT = """\
You are a system improvement agent. Analyze the following execution traces \
and evaluation results, then suggest ONE specific improvement.

Traces summary:
{traces}

Evaluation results:
{eval_results}

Current configuration:
{config}

Suggest an improvement as JSON:
{{
  "target": "prompt|policy|threshold|config",
  "description": "what to change and why",
  "before": "current value (if applicable)",
  "after": "suggested new value"
}}
"""


class MetaAgent:
    """Controlled improvement loop for prompts/policies/config."""

    def __init__(self, llm: LLM, registry_path: str = "improvements") -> None:
        self.llm = llm
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._improvements: list[Improvement] = []

    async def suggest_improvement(
        self,
        traces: str,
        eval_results: str,
        config: str,
    ) -> Improvement:
        prompt = IMPROVEMENT_PROMPT.format(
            traces=traces[:3000],
            eval_results=eval_results[:2000],
            config=config[:1000],
        )

        raw = await self.llm.generate(prompt, mode=LLMMode.THINKING)

        # Parse JSON from response
        try:
            text = raw.strip()
            if "```" in text:
                blocks = text.split("```")
                for block in blocks[1::2]:
                    block = block.strip()
                    if block.startswith("json"):
                        block = block[4:].strip()
                    try:
                        data = json.loads(block)
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    data = json.loads(text[text.find("{"):text.rfind("}") + 1])
            else:
                data = json.loads(text[text.find("{"):text.rfind("}") + 1])
        except (json.JSONDecodeError, ValueError):
            data = {"target": "unknown", "description": raw[:200]}

        improvement = Improvement(
            created_at=datetime.now(timezone.utc).isoformat(),
            **{k: v for k, v in data.items() if k in Improvement.model_fields},
        )
        return improvement

    def register(self, improvement: Improvement) -> None:
        self._improvements.append(improvement)
        path = self.registry_path / f"{len(self._improvements):04d}.json"
        path.write_text(improvement.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Registered improvement: %s", improvement.description[:80])

    def list_improvements(self) -> list[Improvement]:
        return list(self._improvements)

    def rollback(self, improvement_id: str) -> bool:
        """Mark an improvement as rolled back."""
        for imp in self._improvements:
            if imp.improvement_id == improvement_id:
                imp.validated = False
                logger.info("Rolled back improvement: %s", improvement_id)
                return True
        return False
