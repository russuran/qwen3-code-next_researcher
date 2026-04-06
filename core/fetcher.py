"""Fetcher: downloads content via HTTP with optional browser fallback."""
from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FetchResult(BaseModel):
    url: str
    content: str = ""
    status_code: int = 0
    headers: dict[str, str] = {}
    content_type: str = ""
    error: str | None = None


async def fetch(url: str, mode: str = "direct", timeout: int = 30) -> FetchResult:
    """Fetch content from a URL.

    Args:
        url: Target URL.
        mode: 'direct' for httpx, 'browser' for playwright.
        timeout: Request timeout in seconds.
    """
    if mode == "browser":
        return await _fetch_browser(url, timeout)
    return await _fetch_direct(url, timeout)


async def _fetch_direct(url: str, timeout: int) -> FetchResult:
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
            ct = resp.headers.get("content-type", "")
            return FetchResult(
                url=url,
                content=resp.text[:500_000],
                status_code=resp.status_code,
                headers=dict(resp.headers),
                content_type=ct,
            )
    except Exception as e:
        logger.error("Fetch failed for %s: %s", url, e)
        return FetchResult(url=url, error=str(e))


async def _fetch_browser(url: str, timeout: int) -> FetchResult:
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            text = await page.inner_text("body")
            status = resp.status if resp else 0
            await browser.close()
            return FetchResult(
                url=url,
                content=text[:500_000],
                status_code=status,
                content_type="text/html",
            )
    except Exception as e:
        logger.warning("Browser fetch failed for %s: %s, falling back", url, e)
        return await _fetch_direct(url, timeout)
