"""Claim verifier: LLM-as-judge for verifying claims between sources."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from core.llm import LLM, LLMMode

logger = logging.getLogger(__name__)


class ClaimVerification(BaseModel):
    claim: str
    verdict: str = "unverified"  # supported | contradicted | unverified | partially_supported
    confidence: float = 0.0
    supporting_sources: list[str] = []
    contradicting_sources: list[str] = []
    reasoning: str = ""


VERIFY_PROMPT = """\
You are a claim verification judge. Given a claim and a set of source summaries, \
determine if the claim is supported, contradicted, or unverified.

Claim: {claim}

Sources:
{sources}

Respond with ONLY a JSON object:
{{
  "verdict": "supported|contradicted|unverified|partially_supported",
  "confidence": 0.0-1.0,
  "supporting_sources": ["source title that supports"],
  "contradicting_sources": ["source title that contradicts"],
  "reasoning": "one sentence explanation"
}}
"""

CONTRADICTION_PROMPT = """\
You are a contradiction detector. Given the following source analyses, \
identify any contradictions or conflicting claims between them.

Sources:
{sources}

Respond with ONLY a JSON object:
{{
  "contradictions": [
    {{
      "claim_a": "what source A says",
      "source_a": "source A title",
      "claim_b": "what source B says (contradicts A)",
      "source_b": "source B title",
      "severity": "minor|moderate|major"
    }}
  ],
  "consensus": ["claims that all sources agree on"]
}}
"""


class ClaimVerifier:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def verify_claim(
        self, claim: str, sources: list[dict[str, Any]]
    ) -> ClaimVerification:
        sources_text = "\n".join(
            f"- {s.get('title', '')}: {s.get('approach', '')[:200]}"
            for s in sources[:10]
        )
        prompt = VERIFY_PROMPT.format(claim=claim, sources=sources_text)
        raw = await self.llm.generate(prompt, mode=LLMMode.FAST)
        data = self._extract_json(raw)

        return ClaimVerification(
            claim=claim,
            verdict=data.get("verdict", "unverified"),
            confidence=float(data.get("confidence", 0)),
            supporting_sources=data.get("supporting_sources", []),
            contradicting_sources=data.get("contradicting_sources", []),
            reasoning=data.get("reasoning", ""),
        )

    async def detect_contradictions(
        self, sources: list[dict[str, Any]]
    ) -> dict[str, Any]:
        sources_text = "\n".join(
            f"- {s.get('title', '')}: {s.get('approach', '')[:150]}"
            f"\n  Strengths: {', '.join(s.get('strengths', [])[:2])}"
            f"\n  Weaknesses: {', '.join(s.get('weaknesses', [])[:2])}"
            for s in sources[:15]
        )
        prompt = CONTRADICTION_PROMPT.format(sources=sources_text)
        raw = await self.llm.generate(prompt, mode=LLMMode.THINKING)
        data = self._extract_json(raw)
        return {
            "contradictions": data.get("contradictions", []),
            "consensus": data.get("consensus", []),
        }

    @staticmethod
    def _extract_json(text: str) -> dict:
        import json
        text = text.strip()
        if "```" in text:
            for block in text.split("```")[1::2]:
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except Exception:
                    continue
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
        return {}
