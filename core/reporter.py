"""Reporter: formats reports in Markdown, JSON, and HTML."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class Reporter:
    """Converts a Markdown report to multiple output formats."""

    def __init__(self, report_md: str, metadata: dict[str, Any] | None = None) -> None:
        self._md = report_md
        self._metadata = metadata or {}

    def to_markdown(self) -> str:
        return self._md

    def to_json(self) -> str:
        """Convert report to structured JSON."""
        sections = self._parse_sections()
        payload = {
            "metadata": self._metadata,
            "sections": sections,
            "full_text": self._md,
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def to_html(self) -> str:
        """Convert Markdown report to basic HTML."""
        html = self._md

        # Headings
        for level in range(4, 0, -1):
            pattern = re.compile(rf"^{'#' * level}\s+(.+)$", re.MULTILINE)
            html = pattern.sub(rf"<h{level}>\1</h{level}>", html)

        # Code blocks
        html = re.sub(
            r"```(\w*)\n(.*?)```",
            r"<pre><code class='\1'>\2</code></pre>",
            html, flags=re.DOTALL,
        )

        # Inline code
        html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)

        # Bold / italic
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)

        # Line breaks -> paragraphs
        paragraphs = html.split("\n\n")
        html = "\n".join(
            f"<p>{p.strip()}</p>" if not p.strip().startswith("<") else p
            for p in paragraphs if p.strip()
        )

        return f"<!DOCTYPE html>\n<html><head><meta charset='utf-8'></head>\n<body>\n{html}\n</body></html>"

    def _parse_sections(self) -> list[dict[str, str]]:
        sections: list[dict[str, str]] = []
        parts = re.split(r"(^#{1,4}\s+.+$)", self._md, flags=re.MULTILINE)
        current_heading = ""
        for part in parts:
            stripped = part.strip()
            if re.match(r"^#{1,4}\s+", stripped):
                current_heading = re.sub(r"^#+\s*", "", stripped)
            elif stripped:
                sections.append({"heading": current_heading, "content": stripped})
        return sections
