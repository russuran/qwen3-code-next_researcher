"""Parser: splits documents into structured chunks with metadata."""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    chunk_id: str = ""
    text: str
    heading: str = ""
    chunk_type: str = "paragraph"  # paragraph | code | heading | list
    position: int = 0


class ParsedDocument(BaseModel):
    source_url: str = ""
    chunks: list[Chunk] = []
    metadata: dict[str, Any] = {}
    total_chars: int = 0


def parse(text: str, source_url: str = "") -> ParsedDocument:
    """Parse normalized text into structured chunks."""
    doc = ParsedDocument(source_url=source_url, total_chars=len(text))
    sections = _split_sections(text)

    for idx, (heading, body) in enumerate(sections):
        if not body.strip():
            continue
        chunk_id = hashlib.md5(f"{source_url}:{idx}".encode()).hexdigest()[:10]
        chunk_type = _detect_type(body)
        doc.chunks.append(Chunk(
            chunk_id=chunk_id,
            text=body.strip(),
            heading=heading,
            chunk_type=chunk_type,
            position=idx,
        ))

    logger.info("Parsed %d chunks from %s", len(doc.chunks), source_url or "input")
    return doc


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (heading, body) pairs using markdown headings."""
    parts = re.split(r"(^#{1,4}\s+.+$)", text, flags=re.MULTILINE)
    sections: list[tuple[str, str]] = []
    current_heading = ""

    for part in parts:
        stripped = part.strip()
        if re.match(r"^#{1,4}\s+", stripped):
            current_heading = re.sub(r"^#+\s*", "", stripped)
        elif stripped:
            sections.append((current_heading, stripped))

    if not sections and text.strip():
        sections.append(("", text.strip()))

    return sections


def _detect_type(text: str) -> str:
    if text.strip().startswith("```") or text.strip().startswith("    "):
        return "code"
    if re.match(r"^\s*[-*]\s", text):
        return "list"
    return "paragraph"
