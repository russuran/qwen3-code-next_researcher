from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import httpx

# Ensure tools are registered
import core.pdf_parser  # noqa: F401
import core.browser  # noqa: F401


# ---------------------------------------------------------------------------
# download_pdf
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_download_pdf(tmp_path: Path):
    pdf_bytes = b"%PDF-1.4 fake content"
    save_path = str(tmp_path / "papers" / "test.pdf")

    fake_response = httpx.Response(
        200,
        content=pdf_bytes,
        request=httpx.Request("GET", "https://arxiv.org/pdf/1234.pdf"),
    )

    mock_client = AsyncMock()
    mock_client.get.return_value = fake_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("core.pdf_parser.httpx.AsyncClient", return_value=mock_client):
        result = await core.pdf_parser.download_pdf(
            url="https://arxiv.org/pdf/1234.pdf", save_path=save_path
        )

    assert result.success is True
    assert result.data["size_bytes"] == len(pdf_bytes)
    assert Path(save_path).exists()
    assert Path(save_path).read_bytes() == pdf_bytes


# ---------------------------------------------------------------------------
# parse_pdf
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parse_pdf():
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Page 1 content about transformers."

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.close = MagicMock()

    with patch("core.pdf_parser.fitz") as mock_fitz:
        mock_fitz.open.return_value = mock_doc
        result = await core.pdf_parser.parse_pdf(file_path="/fake/paper.pdf")

    assert result.success is True
    assert result.data["num_pages"] == 1
    assert "transformers" in result.data["text"]


@pytest.mark.asyncio
async def test_parse_pdf_truncation():
    long_text = "word " * 5000  # ~25000 chars
    mock_page = MagicMock()
    mock_page.get_text.return_value = long_text

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.close = MagicMock()

    with patch("core.pdf_parser.fitz") as mock_fitz:
        mock_fitz.open.return_value = mock_doc
        result = await core.pdf_parser.parse_pdf(file_path="/fake/paper.pdf", max_chars=100)

    assert result.success is True
    assert "[truncated]" in result.data["text"]


# ---------------------------------------------------------------------------
# parse_latex
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parse_latex():
    result = await core.pdf_parser.parse_latex(text=r"\textbf{Hello} $x^2$")
    assert result.success is True
    assert "Hello" in result.data["text"]


# ---------------------------------------------------------------------------
# inspect_code
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inspect_code():
    fake_response = httpx.Response(
        200,
        text="# My Project\n\nThis is a README.",
        request=httpx.Request("GET", "https://raw.githubusercontent.com/user/repo/main/README.md"),
    )

    mock_client = AsyncMock()
    mock_client.get.return_value = fake_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("core.pdf_parser.httpx.AsyncClient", return_value=mock_client):
        result = await core.pdf_parser.inspect_code(url="https://github.com/user/repo")

    assert result.success is True
    assert "My Project" in result.data["content"]


# ---------------------------------------------------------------------------
# browse_web (httpx fallback only — no Playwright in tests)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_browse_web_httpx_fallback():
    html = "<html><body><h1>Title</h1><p>Content here</p><script>var x=1;</script></body></html>"
    fake_response = httpx.Response(
        200,
        text=html,
        request=httpx.Request("GET", "https://example.com"),
    )

    mock_client = AsyncMock()
    mock_client.get.return_value = fake_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    # Force fallback mode
    original_browser = core.browser._browser
    core.browser._browser = False

    try:
        with patch("core.browser.httpx.AsyncClient", return_value=mock_client):
            result = await core.browser.browse_web(url="https://example.com")
    finally:
        core.browser._browser = original_browser

    assert result.success is True
    assert "Title" in result.data["text"]
    assert "Content here" in result.data["text"]
    # Script content should be stripped
    assert "var x=1" not in result.data["text"]
