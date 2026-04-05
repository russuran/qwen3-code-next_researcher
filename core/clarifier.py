"""Clarifier: detects ambiguous queries and generates clarification questions."""
from __future__ import annotations

import logging

from pydantic import BaseModel

from core.llm import LLM, LLMMode

logger = logging.getLogger(__name__)


class ClarificationResult(BaseModel):
    needs_clarification: bool = False
    questions: list[str] = []
    confidence: float = 1.0
    suggested_refinement: str = ""


CLARIFY_PROMPT = """\
You are a research query analyst. Determine if the following query is clear enough \
to start a research investigation, or if clarification is needed.

Query: {query}

Respond with ONLY a JSON object:
{{
  "needs_clarification": true/false,
  "confidence": 0.0-1.0 (how confident you are the query is clear),
  "questions": ["clarification question 1", "question 2"],
  "suggested_refinement": "a more specific version of the query if ambiguous"
}}

A query needs clarification if:
- It's too vague (e.g., "tell me about AI")
- It has multiple possible interpretations
- Critical constraints are missing (language, platform, year range)
- The scope is unclear

A query does NOT need clarification if:
- It names specific tools/methods (e.g., "LangChain vs CrewAI for ReAct")
- It has clear scope and expected output
"""


class Clarifier:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def check(self, query: str) -> ClarificationResult:
        prompt = CLARIFY_PROMPT.format(query=query)
        raw = await self.llm.generate(prompt, mode=LLMMode.FAST)

        data = self._extract_json(raw)
        return ClarificationResult(
            needs_clarification=data.get("needs_clarification", False),
            questions=data.get("questions", []),
            confidence=float(data.get("confidence", 1.0)),
            suggested_refinement=data.get("suggested_refinement", ""),
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        import json
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
        return {}
