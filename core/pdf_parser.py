from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF
import httpx
from pylatexenc.latex2text import LatexNodes2Text

from core.tools import ToolParam, ToolResult, registry

logger = logging.getLogger(__name__)


@registry.register(
    name="download_pdf",
    description="Download a PDF file from a URL and save it locally.",
    params=[
        ToolParam(name="url", type="str", description="PDF URL to download"),
        ToolParam(name="save_path", type="str", description="Local path to save the file"),
    ],
)
async def download_pdf(url: str, save_path: str) -> ToolResult:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    path.write_bytes(resp.content)
    logger.info("Downloaded PDF: %s -> %s (%d bytes)", url, save_path, len(resp.content))

    return ToolResult(
        tool_name="download_pdf",
        success=True,
        data={"path": str(path), "size_bytes": len(resp.content)},
    )


@registry.register(
    name="parse_pdf",
    description="Extract text from a PDF file. Returns the full text content.",
    params=[
        ToolParam(name="file_path", type="str", description="Path to the PDF file"),
        ToolParam(name="max_chars", type="int", description="Max characters to extract", required=False, default=12000),
    ],
)
async def parse_pdf(file_path: str, max_chars: int = 12000) -> ToolResult:
    doc = fitz.open(file_path)
    pages_text: list[str] = []

    for page in doc:
        pages_text.append(page.get_text())

    doc.close()

    full_text = "\n\n".join(pages_text)
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n... [truncated]"

    return ToolResult(
        tool_name="parse_pdf",
        success=True,
        data={"file_path": file_path, "num_pages": len(pages_text), "text": full_text},
    )


@registry.register(
    name="parse_latex",
    description="Convert LaTeX markup to plain text.",
    params=[
        ToolParam(name="text", type="str", description="LaTeX text to convert"),
    ],
)
async def parse_latex(text: str) -> ToolResult:
    converter = LatexNodes2Text()
    plain = converter.latex_to_text(text)

    return ToolResult(tool_name="parse_latex", success=True, data={"text": plain})


@registry.register(
    name="inspect_code",
    description="Fetch and return source code from a GitHub URL (raw file or README).",
    params=[
        ToolParam(name="url", type="str", description="GitHub repository URL"),
        ToolParam(name="path", type="str", description="File path within the repo (defaults to README)", required=False, default=None),
    ],
)
async def inspect_code(url: str, path: str | None = None) -> ToolResult:
    # Convert github.com URL to raw content URL
    # e.g. https://github.com/user/repo -> raw README
    # e.g. https://github.com/user/repo with path=src/main.py -> raw file
    raw_url = url.replace("github.com", "raw.githubusercontent.com")

    if "/tree/" not in raw_url and "/blob/" not in raw_url:
        # Assume main branch
        branch = "main"
        file_path = path or "README.md"
        raw_url = f"{raw_url}/{branch}/{file_path}"
    else:
        raw_url = raw_url.replace("/blob/", "/").replace("/tree/", "/")
        if path:
            raw_url = f"{raw_url}/{path}"

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(raw_url)
        if resp.status_code == 404 and "main" in raw_url:
            # Try master branch as fallback
            resp = await client.get(raw_url.replace("/main/", "/master/"))
        resp.raise_for_status()

    content = resp.text[:10000]

    return ToolResult(
        tool_name="inspect_code",
        success=True,
        data={"url": raw_url, "content": content},
    )
