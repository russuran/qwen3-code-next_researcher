from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from pydantic import BaseModel
from slugify import slugify

from core.llm import LLM, LLMMode
from core.prompts import PLAN_GENERATION, SEARCH_REFINEMENT

logger = logging.getLogger(__name__)


class SubQuestion(BaseModel):
    question: str
    priority: int = 3
    sources: list[str] = ["arxiv", "github"]
    keywords: list[str] = []


class ResearchPlan(BaseModel):
    topic: str
    slug: str = ""
    sub_questions: list[SubQuestion] = []
    scope_notes: str = ""


class RefinedQuery(BaseModel):
    query: str
    source: str
    reason: str


class SearchRefinement(BaseModel):
    refined_queries: list[RefinedQuery] = []


class Planner:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    @staticmethod
    def _inject_date(text: str) -> str:
        now = datetime.now(timezone.utc)
        return (
            text
            .replace("{{ current_date }}", now.strftime("%Y-%m-%d"))
            .replace("{{ current_year }}", str(now.year))
        )

    async def generate_plan(self, topic: str) -> ResearchPlan:
        prompt = self._inject_date(PLAN_GENERATION.replace("{{ topic }}", topic))

        raw = await self.llm.generate(prompt, mode=LLMMode.THINKING)

        # Extract JSON from response
        plan_data = self._extract_json(raw)

        plan = ResearchPlan(
            topic=topic,
            slug=slugify(topic, max_length=60),
            sub_questions=[SubQuestion(**sq) for sq in plan_data.get("sub_questions", [])],
            scope_notes=plan_data.get("scope_notes", ""),
        )

        logger.info(
            "Generated plan for '%s': %d sub-questions",
            topic, len(plan.sub_questions),
        )
        return plan

    async def refine_plan(
        self,
        plan: ResearchPlan,
        findings_summary: str,
    ) -> list[RefinedQuery]:
        prompt = (
            SEARCH_REFINEMENT
            .replace("{{ topic }}", plan.topic)
            .replace("{{ findings_summary }}", findings_summary)
        )

        raw = await self.llm.generate(prompt, mode=LLMMode.THINKING)
        data = self._extract_json(raw)

        queries = [RefinedQuery(**q) for q in data.get("refined_queries", [])]
        logger.info("Refined plan: %d additional queries", len(queries))
        return queries

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = text.strip()

        # Try to find JSON in markdown code block
        if "```" in text:
            blocks = text.split("```")
            for block in blocks[1::2]:  # odd-indexed = inside code blocks
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue

        # Try the whole text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find first { ... } block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse JSON from LLM response, returning empty dict")
        return {}
