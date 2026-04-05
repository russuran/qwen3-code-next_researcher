from __future__ import annotations

import logging
from html.parser import HTMLParser

import httpx

from core.tools import ToolParam, ToolResult, registry

logger = logging.getLogger(__name__)

# Singleton browser instance (lazy-initialized)
_browser = None
_playwright_ctx = None


async def _get_browser():
    global _browser, _playwright_ctx
    if _browser is None:
        try:
            from playwright.async_api import async_playwright
            _playwright_ctx = await async_playwright().start()
            _browser = await _playwright_ctx.chromium.launch(headless=True)
            logger.info("Playwright browser launched")
        except Exception as e:
            logger.warning("Playwright unavailable (%s), will use httpx fallback", e)
            _browser = False  # sentinel: fallback mode
    return _browser


class _TextExtractor(HTMLParser):
    """Simple HTML-to-text extractor as Playwright fallback."""

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


async def _browse_with_playwright(url: str, max_chars: int) -> str:
    browser = await _get_browser()
    page = await browser.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        text = await page.inner_text("body")
        return text[:max_chars]
    finally:
        await page.close()


async def _browse_with_httpx(url: str, max_chars: int) -> str:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    extractor = _TextExtractor()
    extractor.feed(resp.text)
    return extractor.get_text()[:max_chars]


@registry.register(
    name="browse_web",
    description="Browse a web page and extract its text content.",
    params=[
        ToolParam(name="url", type="str", description="URL to browse"),
        ToolParam(name="max_chars", type="int", description="Max characters to extract", required=False, default=8000),
    ],
)
async def browse_web(url: str, max_chars: int = 8000) -> ToolResult:
    browser = await _get_browser()

    if browser and browser is not False:
        try:
            text = await _browse_with_playwright(url, max_chars)
            return ToolResult(tool_name="browse_web", success=True, data={"url": url, "text": text})
        except Exception as e:
            logger.warning("Playwright failed for %s: %s, falling back to httpx", url, e)

    text = await _browse_with_httpx(url, max_chars)
    return ToolResult(tool_name="browse_web", success=True, data={"url": url, "text": text})


async def close_browser() -> None:
    global _browser, _playwright_ctx
    if _browser and _browser is not False:
        await _browser.close()
    if _playwright_ctx:
        await _playwright_ctx.stop()
    _browser = None
    _playwright_ctx = None
