"""Extractor: pulls claims, facts, metrics, and code from parsed documents via LLM."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from core.llm import LLM, LLMMode
from core.parser import ParsedDocument

logger = logging.getLogger(__name__)


class Extraction(BaseModel):
    source_url: str = ""
    claims: list[str] = []
    facts: list[str] = []
    metrics: list[dict[str, Any]] = []
    code_snippets: list[str] = []


_EXTRACT_PROMPT = """\
Extract structured information from the following document.

Source: {source}
Document text (truncated):
{text}

Return ONLY a JSON object:
{{
  "claims": ["specific claims made in the text"],
  "facts": ["objective factual statements"],
  "metrics": [{{"name": "metric name", "value": "value", "context": "..."}}],
  "code_snippets": ["relevant code or command examples"]
}}
"""


async def extract(doc: ParsedDocument, llm: LLM) -> Extraction:
    """Extract structured information from a parsed document."""
    text = "\n\n".join(c.text for c in doc.chunks)[:6000]
    prompt = _EXTRACT_PROMPT.format(source=doc.source_url, text=text)

    try:
        result = await llm.generate_structured(
            prompt, Extraction, mode=LLMMode.FAST,
        )
        result.source_url = doc.source_url  # type: ignore[union-attr]
        logger.info(
            "Extracted from %s: %d claims, %d facts, %d metrics",
            doc.source_url,
            len(result.claims),  # type: ignore[union-attr]
            len(result.facts),  # type: ignore[union-attr]
            len(result.metrics),  # type: ignore[union-attr]
        )
        return result  # type: ignore[return-value]
    except Exception as e:
        logger.error("Extraction failed for %s: %s", doc.source_url, e)
        return Extraction(source_url=doc.source_url)
