"""Contradiction detector: standalone detection of contradictions across sources."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from core.llm import LLM, LLMMode

logger = logging.getLogger(__name__)


class Contradiction(BaseModel):
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    severity: str = "moderate"  # minor | moderate | major


_DETECT_PROMPT = """\
Identify contradictions between these sources. Sources:
{sources}
Return ONLY a JSON array: [{{"claim_a": "...", "source_a": "...", \
"claim_b": "...", "source_b": "...", "severity": "minor|moderate|major"}}]
Return [] if none found.
"""


class ContradictionDetector:
    """Detects contradictions between multiple source analyses using LLM."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def detect(self, sources: list[dict[str, Any]]) -> list[Contradiction]:
        """Detect contradictions across source analyses."""
        if len(sources) < 2:
            return []

        sources_text = "\n".join(
            f"- {s.get('title', 'Unknown')}: {str(s.get('approach', s.get('summary', '')))[:200]}"
            for s in sources[:15]
        )
        prompt = _DETECT_PROMPT.format(sources=sources_text)

        try:
            raw = await self.llm.generate(prompt, mode=LLMMode.THINKING)
            data = self._parse_list(raw)
            contradictions = [Contradiction(**item) for item in data if isinstance(item, dict)]
            logger.info("Detected %d contradictions across %d sources", len(contradictions), len(sources))
            return contradictions
        except Exception as e:
            logger.error("Contradiction detection failed: %s", e)
            return []

    @staticmethod
    def _parse_list(text: str) -> list[dict[str, Any]]:
        import json
        text = text.strip()
        if "```" in text:
            for block in text.split("```")[1::2]:
                block = block.strip().removeprefix("json").strip()
                try:
                    return json.loads(block)
                except Exception:
                    continue
        start, end = text.find("["), text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
        return []
