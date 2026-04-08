from __future__ import annotations

import asyncio
import logging
import os

import httpx

from core.tools import ToolParam, ToolResult, registry

logger = logging.getLogger(__name__)

# Delay between API calls to avoid rate limits
_RATE_LIMIT_DELAY = 1.0  # seconds


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

@registry.register(
    name="search_arxiv",
    description="Search arXiv for academic papers. Returns titles, abstracts, PDF URLs.",
    params=[
        ToolParam(name="query", type="str", description="Search query"),
        ToolParam(name="max_results", type="int", description="Max results to return", required=False, default=10),
    ],
)
async def search_arxiv(query: str, max_results: int = 10) -> ToolResult:
    import arxiv

    def _search() -> list[dict]:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        results = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "authors": [a.name for a in paper.authors[:5]],
                "abstract": paper.summary[:500],
                "pdf_url": paper.pdf_url,
                "arxiv_id": paper.entry_id.split("/")[-1],
                "published": paper.published.isoformat() if paper.published else None,
                "categories": paper.categories,
            })
        return results

    data = await asyncio.to_thread(_search)
    return ToolResult(tool_name="search_arxiv", success=True, data=data)


# ---------------------------------------------------------------------------
# Semantic Scholar (httpx, no SDK — avoids event loop issues)
# ---------------------------------------------------------------------------

_S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_S2_FIELDS = "title,authors,abstract,citationCount,year,url,externalIds"

@registry.register(
    name="search_semantic_scholar",
    description="Search Semantic Scholar for academic papers with citation counts.",
    params=[
        ToolParam(name="query", type="str", description="Search query"),
        ToolParam(name="max_results", type="int", description="Max results to return", required=False, default=10),
    ],
)
async def search_semantic_scholar(query: str, max_results: int = 10) -> ToolResult:
    await asyncio.sleep(_RATE_LIMIT_DELAY)

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            _S2_API,
            params={"query": query, "limit": max_results, "fields": _S2_FIELDS},
        )
        if resp.status_code == 429:
            logger.warning("Semantic Scholar rate limited, waiting 5s...")
            await asyncio.sleep(5)
            resp = await client.get(
                _S2_API,
                params={"query": query, "limit": max_results, "fields": _S2_FIELDS},
            )
        resp.raise_for_status()
        raw = resp.json()

    results = []
    for paper in raw.get("data", []):
        authors = paper.get("authors") or []
        results.append({
            "title": paper.get("title", ""),
            "authors": [a.get("name", "") for a in authors[:5]],
            "abstract": (paper.get("abstract") or "")[:500],
            "citation_count": paper.get("citationCount", 0),
            "year": paper.get("year"),
            "url": paper.get("url", ""),
            "external_ids": paper.get("externalIds") or {},
        })

    return ToolResult(tool_name="search_semantic_scholar", success=True, data=results)


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------

@registry.register(
    name="search_github",
    description="Search GitHub repositories by topic. Returns name, description, stars, URL.",
    params=[
        ToolParam(name="query", type="str", description="Search query"),
        ToolParam(name="max_results", type="int", description="Max results to return", required=False, default=10),
    ],
)
async def search_github(query: str, max_results: int = 10) -> ToolResult:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": max_results},
            headers=headers,
        )
        resp.raise_for_status()
        raw = resp.json()

    results = []
    for item in raw.get("items", []):
        results.append({
            "name": item["full_name"],
            "description": (item.get("description") or "")[:300],
            "stars": item.get("stargazers_count", 0),
            "language": item.get("language"),
            "url": item["html_url"],
            "topics": item.get("topics", []),
        })

    return ToolResult(tool_name="search_github", success=True, data=results)


# ---------------------------------------------------------------------------
# Papers With Code
# ---------------------------------------------------------------------------

@registry.register(
    name="search_papers_with_code",
    description="Search Papers With Code for ML methods with linked source code.",
    params=[
        ToolParam(name="query", type="str", description="Search query"),
        ToolParam(name="max_results", type="int", description="Max results to return", required=False, default=10),
    ],
)
async def search_papers_with_code(query: str, max_results: int = 10) -> ToolResult:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(
            "https://paperswithcode.com/api/v1/search/",
            params={"q": query, "page": 1, "items_per_page": max_results},
        )
        resp.raise_for_status()
        raw = resp.json()

    results = []
    for item in raw.get("results", []):
        results.append({
            "title": item.get("paper", {}).get("title", ""),
            "abstract": (item.get("paper", {}).get("abstract", "") or "")[:500],
            "paper_url": item.get("paper", {}).get("url_abs", ""),
            "repository_url": item.get("repository", {}).get("url", ""),
        })

    return ToolResult(tool_name="search_papers_with_code", success=True, data=results)


# ---------------------------------------------------------------------------
# HuggingFace Hub (models + datasets)
# ---------------------------------------------------------------------------

@registry.register(
    name="search_huggingface",
    description="Search HuggingFace Hub for models and datasets. Supports language filter.",
    params=[
        ToolParam(name="query", type="str", description="Search query"),
        ToolParam(name="max_results", type="int", description="Max results", required=False, default=10),
        ToolParam(name="resource_type", type="str", description="'models' or 'datasets'", required=False, default="models"),
    ],
)
async def search_huggingface(query: str, max_results: int = 10, resource_type: str = "models") -> ToolResult:
    endpoint = f"https://huggingface.co/api/{resource_type}"
    params = {"search": query, "limit": max_results, "sort": "downloads", "direction": "-1"}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(endpoint, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return ToolResult(tool_name="search_huggingface", success=False, error=str(e))

    results = []
    for item in data[:max_results]:
        item_id = item.get("id") or item.get("modelId", "")
        results.append({
            "name": item_id,
            "title": item_id,
            "url": f"https://huggingface.co/{item_id}",
            "description": (item.get("description") or item.get("cardData", {}).get("description", ""))[:300],
            "downloads": item.get("downloads", 0),
            "likes": item.get("likes", 0),
            "tags": item.get("tags", [])[:5],
        })

    return ToolResult(tool_name="search_huggingface", success=True, data=results)
