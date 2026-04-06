"""Synthesizer: generates final deliverables from analysis results."""
from __future__ import annotations

import logging
from typing import Any

from core.analyzer import AnalysisResult
from core.llm import LLM, LLMMode

logger = logging.getLogger(__name__)

_SYNTHESIS_PROMPT = """\
You are a technical writer producing a comprehensive research report.

Topic: {topic}

Analysis summary: {summary}

Themes: {themes}

Consensus points: {consensus}

Contradictions: {contradictions}

Key metrics: {metrics}

Write a well-structured Markdown report with:
1. Executive Summary
2. Key Findings (organized by theme)
3. Methodology Comparison (if applicable)
4. Metrics & Benchmarks
5. Contradictions & Open Questions
6. Recommendations
7. References

Use proper Markdown formatting with headings, bullet points, and code blocks \
where appropriate.
"""


async def synthesize(
    analysis: AnalysisResult,
    plan: dict[str, Any],
    llm: LLM,
) -> str:
    """Generate a final Markdown report from analysis results."""
    prompt = _SYNTHESIS_PROMPT.format(
        topic=plan.get("topic", "Research Report"),
        summary=analysis.summary,
        themes=", ".join(analysis.themes),
        consensus="\n".join(f"- {c}" for c in analysis.consensus),
        contradictions="\n".join(
            f"- {c.get('claim_a', '')} vs {c.get('claim_b', '')}"
            for c in analysis.contradictions
        ),
        metrics="\n".join(
            f"- {m.get('name', '')}: {m.get('value', '')}"
            for m in analysis.key_metrics
        ),
    )

    try:
        report = await llm.generate(prompt, mode=LLMMode.THINKING)
        logger.info("Synthesized report: %d chars", len(report))
        return report
    except Exception as e:
        logger.error("Synthesis failed: %s", e)
        return f"# Report Generation Failed\n\nError: {e}\n\n## Raw Summary\n\n{analysis.summary}"
