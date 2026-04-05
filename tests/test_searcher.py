from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

# Ensure tools are registered by importing the modules
import core.searcher  # noqa: F401


# ---------------------------------------------------------------------------
# search_arxiv
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_arxiv():
    mock_paper = MagicMock()
    mock_paper.title = "Attention Is All You Need"
    mock_paper.summary = "We propose a new architecture..."
    mock_paper.pdf_url = "https://arxiv.org/pdf/1706.03762"
    mock_paper.entry_id = "http://arxiv.org/abs/1706.03762"
    mock_paper.published = None
    mock_paper.categories = ["cs.CL"]
    mock_author = MagicMock()
    mock_author.name = "Vaswani"
    mock_paper.authors = [mock_author]

    mock_client = MagicMock()
    mock_client.results.return_value = [mock_paper]

    with patch("arxiv.Client", return_value=mock_client), \
         patch("arxiv.Search") as mock_search, \
         patch("arxiv.SortCriterion") as mock_sort:
        mock_sort.Relevance = "relevance"

        result = await core.searcher.search_arxiv(query="transformers", max_results=1)

    assert result.success is True
    assert len(result.data) == 1
    assert result.data[0]["title"] == "Attention Is All You Need"
    assert result.data[0]["authors"] == ["Vaswani"]


# ---------------------------------------------------------------------------
# search_semantic_scholar
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_semantic_scholar():
    fake_response = httpx.Response(
        200,
        json={
            "data": [
                {
                    "title": "BERT: Pre-training",
                    "authors": [{"name": "Devlin"}],
                    "abstract": "We introduce BERT...",
                    "citationCount": 50000,
                    "year": 2019,
                    "url": "https://semanticscholar.org/paper/123",
                    "externalIds": {"ArXiv": "1810.04805"},
                }
            ]
        },
        request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
    )

    mock_client = AsyncMock()
    mock_client.get.return_value = fake_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("core.searcher.httpx.AsyncClient", return_value=mock_client):
        result = await core.searcher.search_semantic_scholar(query="BERT", max_results=1)

    assert result.success is True
    assert len(result.data) == 1
    assert result.data[0]["title"] == "BERT: Pre-training"
    assert result.data[0]["citation_count"] == 50000


# ---------------------------------------------------------------------------
# search_github (httpx mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_github():
    fake_response = httpx.Response(
        200,
        json={
            "items": [
                {
                    "full_name": "huggingface/transformers",
                    "description": "State-of-the-art ML",
                    "stargazers_count": 120000,
                    "language": "Python",
                    "html_url": "https://github.com/huggingface/transformers",
                    "topics": ["nlp", "transformers"],
                }
            ]
        },
        request=httpx.Request("GET", "https://api.github.com/search/repositories"),
    )

    mock_client = AsyncMock()
    mock_client.get.return_value = fake_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("core.searcher.httpx.AsyncClient", return_value=mock_client):
        result = await core.searcher.search_github(query="transformers", max_results=1)

    assert result.success is True
    assert len(result.data) == 1
    assert result.data[0]["name"] == "huggingface/transformers"
    assert result.data[0]["stars"] == 120000


# ---------------------------------------------------------------------------
# search_papers_with_code (httpx mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_papers_with_code():
    fake_response = httpx.Response(
        200,
        json={
            "results": [
                {
                    "paper": {
                        "title": "ResNet",
                        "abstract": "Deep residual learning...",
                        "url_abs": "https://paperswithcode.com/paper/resnet",
                    },
                    "repository": {
                        "url": "https://github.com/pytorch/vision",
                    },
                }
            ]
        },
        request=httpx.Request("GET", "https://paperswithcode.com/api/v1/search/"),
    )

    mock_client = AsyncMock()
    mock_client.get.return_value = fake_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("core.searcher.httpx.AsyncClient", return_value=mock_client):
        result = await core.searcher.search_papers_with_code(query="resnet", max_results=1)

    assert result.success is True
    assert len(result.data) == 1
    assert result.data[0]["title"] == "ResNet"
    assert "pytorch" in result.data[0]["repository_url"]
