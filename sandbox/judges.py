"""LLM-as-judge evaluators for content quality assessment."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from core.llm import LLM, LLMMode

logger = logging.getLogger(__name__)


class JudgeResult(BaseModel):
    score: float = 0.0
    reasoning: str = ""
    criteria_scores: dict[str, float] = {}


_JUDGE_PROMPT = """\
You are an expert judge evaluating content quality.

Content to evaluate:
{content}

Evaluation criteria:
{criteria}

Score each criterion from 0.0 to 1.0, then provide an overall score.

Return ONLY a JSON object:
{{
  "score": 0.0,
  "reasoning": "brief explanation",
  "criteria_scores": {{"criterion_name": 0.0}}
}}
"""


class LLMJudge:
    """Uses an LLM to evaluate content against specified criteria."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def judge(
        self,
        content: str,
        criteria: list[str],
    ) -> JudgeResult:
        """Evaluate content against criteria using LLM-as-judge."""
        criteria_text = "\n".join(f"- {c}" for c in criteria)
        prompt = _JUDGE_PROMPT.format(
            content=content[:6000],
            criteria=criteria_text,
        )

        try:
            result = await self.llm.generate_structured(
                prompt, JudgeResult, mode=LLMMode.THINKING,
            )
            logger.info("Judge score: %.2f (%d criteria)", result.score, len(result.criteria_scores))  # type: ignore[union-attr]
            return result  # type: ignore[return-value]
        except Exception as e:
            logger.error("Judge evaluation failed: %s", e)
            return JudgeResult(reasoning=f"Evaluation failed: {e}")
