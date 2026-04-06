"""Analyzer: cross-source analysis and synthesis of extractions."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from core.extractor import Extraction
from core.llm import LLM, LLMMode

logger = logging.getLogger(__name__)


class AnalysisResult(BaseModel):
    themes: list[str] = []
    contradictions: list[dict[str, str]] = []
    consensus: list[str] = []
    key_metrics: list[dict[str, Any]] = []
    summary: str = ""


_ANALYSIS_PROMPT = """\
Analyze the following extractions from multiple sources and identify themes, \
contradictions, and consensus across them.

Sources ({count}):
{sources}

Return ONLY a JSON object:
{{
  "themes": ["major themes across sources"],
  "contradictions": [{{"claim_a": "...", "source_a": "...", "claim_b": "...", "source_b": "..."}}],
  "consensus": ["points all sources agree on"],
  "key_metrics": [{{"name": "...", "value": "...", "source": "..."}}],
  "summary": "2-3 sentence synthesis"
}}
"""


async def analyze(extractions: list[Extraction], llm: LLM) -> AnalysisResult:
    """Cross-reference and analyze multiple source extractions."""
    if not extractions:
        return AnalysisResult(summary="No extractions to analyze.")

    source_texts: list[str] = []
    for ext in extractions[:15]:
        claims_str = "; ".join(ext.claims[:5])
        facts_str = "; ".join(ext.facts[:5])
        source_texts.append(
            f"- {ext.source_url}:\n  Claims: {claims_str}\n  Facts: {facts_str}"
        )

    prompt = _ANALYSIS_PROMPT.format(
        count=len(extractions),
        sources="\n".join(source_texts),
    )

    try:
        result = await llm.generate_structured(
            prompt, AnalysisResult, mode=LLMMode.THINKING,
        )
        logger.info(
            "Analysis: %d themes, %d contradictions, %d consensus",
            len(result.themes),  # type: ignore[union-attr]
            len(result.contradictions),  # type: ignore[union-attr]
            len(result.consensus),  # type: ignore[union-attr]
        )
        return result  # type: ignore[return-value]
    except Exception as e:
        logger.error("Analysis failed: %s", e)
        return AnalysisResult(summary=f"Analysis failed: {e}")
