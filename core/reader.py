"""Reader: normalizes content from different document formats to plain text."""
from __future__ import annotations

import json
import logging
import re
from html.parser import HTMLParser

logger = logging.getLogger(__name__)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            text = data.strip()
            if text:
                self._chunks.append(text)

    def get_text(self) -> str:
        return "\n".join(self._chunks)


def read(content: str, content_type: str = "text/plain") -> str:
    """Read and normalize content based on content_type.

    Supports: HTML, JSON, Markdown, PDF text, plain text.
    """
    ct = content_type.lower()

    if "html" in ct:
        return _read_html(content)
    if "json" in ct:
        return _read_json(content)
    if "pdf" in ct:
        return _read_pdf_text(content)
    if "markdown" in ct or content_type.endswith(".md"):
        return _read_markdown(content)
    return content.strip()


def _read_html(content: str) -> str:
    extractor = _HTMLTextExtractor()
    extractor.feed(content)
    return extractor.get_text()


def _read_json(content: str) -> str:
    try:
        data = json.loads(content)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return content.strip()


def _read_markdown(content: str) -> str:
    # Strip markdown image syntax, keep link text
    text = re.sub(r"!\[.*?\]\(.*?\)", "", content)
    text = re.sub(r"\[([^\]]+)\]\(.*?\)", r"\1", text)
    # Strip HTML tags that sometimes appear in markdown
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _read_pdf_text(content: str) -> str:
    # PDF binary content should be handled by pdf_parser;
    # this handles already-extracted text from PDFs
    return content.strip()
