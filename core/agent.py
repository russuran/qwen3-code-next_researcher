from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel

from core.llm import LLM, LLMMode
from core.planner import Planner, ResearchPlan, SubQuestion
from core.prompts import (
    SYSTEM_REACT, ANALYSIS, DEEP_ANALYSIS, SYNTHESIS, COMPARISON_TABLE,
    RELEVANCE_FILTER, HYPOTHESIS_GENERATION,
)
from core.tools import ToolRegistry, ToolResult, registry
from core.claim_verifier import ClaimVerifier
from knowledge.source_registry import SourceRegistry
from knowledge.document_store import Document, DocumentStore
from knowledge.cache_manager import CacheManager
from knowledge.index_manager import IndexManager
from knowledge.change_detector import ChangeDetector
from knowledge.refresh_scheduler import RefreshScheduler
from sandbox.evaluator import Evaluator

# Import tool modules so they register themselves
import core.searcher  # noqa: F401
import core.browser  # noqa: F401
import core.pdf_parser  # noqa: F401

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

class JournalEntry(BaseModel):
    timestamp: str
    phase: Literal["plan", "search", "recall", "analyze", "synthesize"]
    action: str
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    result_summary: str = ""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Finding(BaseModel):
    source: str
    query: str
    results: list[dict[str, Any]] = []


class AnalysisResult(BaseModel):
    title: str = ""
    approach: str = ""
    key_contributions: list[str] = []
    strengths: list[str] = []
    weaknesses: list[str] = []
    relevant_code: str = ""
    tags: list[str] = []
    raw_source: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# compare_methods tool (LLM-driven)
# ---------------------------------------------------------------------------

_llm_ref: LLM | None = None


@registry.register(
    name="compare_methods",
    description="Build a comparison table for a list of methods/approaches.",
    params=[],
)
async def compare_methods(methods_json: str) -> ToolResult:
    if _llm_ref is None:
        return ToolResult(tool_name="compare_methods", success=False, error="LLM not initialized")
    prompt = COMPARISON_TABLE.replace("{{ methods }}", methods_json)
    table_md = await _llm_ref.generate(prompt, mode=LLMMode.FAST)
    return ToolResult(tool_name="compare_methods", success=True, data={"table": table_md})


# ---------------------------------------------------------------------------
# ReAct parser
# ---------------------------------------------------------------------------

_RE_ACTION = re.compile(r"Action:\s*(\S+)", re.IGNORECASE)
_RE_ACTION_INPUT = re.compile(r"Action Input:\s*(\{.*\})", re.IGNORECASE | re.DOTALL)
_RE_FINAL_ANSWER = re.compile(r"Final Answer:\s*(.*)", re.IGNORECASE | re.DOTALL)


def _parse_react(text: str) -> tuple[str | None, dict | None, str | None]:
    """Parse ReAct output -> (action_name, action_input, final_answer)."""
    final = _RE_FINAL_ANSWER.search(text)
    if final:
        return None, None, final.group(1).strip()

    action_match = _RE_ACTION.search(text)
    input_match = _RE_ACTION_INPUT.search(text)

    if action_match:
        action_name = action_match.group(1).strip()
        action_input = {}
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                logger.warning("Failed to parse Action Input JSON")
        return action_name, action_input, None

    return None, None, None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_ALL_STAGES = ["plan", "search", "filter", "deep_fetch", "analyze", "iterative", "hypotheses", "contradictions", "synthesize"]


class AgentConfig(BaseModel):
    output_dir: str = "./output"
    journal_dir: str = "./journal"
    knowledge_dir: str = "./knowledge_store"
    sources: list[str] = ["arxiv", "semantic_scholar", "github", "papers_with_code"]
    max_results_per_source: int = 20
    parallel_search: bool = True
    verbose: bool = False
    intervene: bool = False
    stages: list[str] = _ALL_STAGES


class ResearchAgent:
    def __init__(
        self,
        config: AgentConfig,
        llm: LLM,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.registry = tool_registry or registry
        self.planner = Planner(llm)
        self._journal_path: Path | None = None

        # Knowledge layer — persistent across runs via knowledge_dir
        knowledge_dir = Path(config.knowledge_dir)
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        self._knowledge_dir = knowledge_dir

        self.source_registry = SourceRegistry()
        self.doc_store = DocumentStore()
        self.cache = CacheManager(cache_dir=str(knowledge_dir / "cache"))
        self.index = IndexManager(self.doc_store)
        self.change_detector = ChangeDetector(self.source_registry)
        self.refresh_scheduler = RefreshScheduler(
            self.source_registry, self.change_detector, self.cache,
        )
        self.evaluator = Evaluator()
        self.claim_verifier = ClaimVerifier(llm)

        # Load persisted knowledge from previous runs
        self.doc_store.load(knowledge_dir / "documents.json")
        self.source_registry.load(knowledge_dir / "sources.json")
        if self.doc_store.count() > 0:
            self.index.build()
            logger.info("Loaded knowledge: %d docs, %d sources, %d chunks",
                        self.doc_store.count(), self.source_registry.count(),
                        self.doc_store.total_chunks())

        # Set global LLM ref for compare_methods tool
        global _llm_ref
        _llm_ref = llm

    def _stage_enabled(self, stage: str) -> bool:
        return stage in self.config.stages

    async def run(self, topic: str) -> Path:
        """Main entry point. Returns path to the output directory."""
        stages = self.config.stages
        console.print(Panel(f"[bold]Deep Research:[/bold] {topic}\n[dim]Stages: {', '.join(stages)}[/dim]", style="blue"))

        # Phase 1: Plan (always runs)
        plan = await self._plan(topic)
        output_dir = Path(self.config.output_dir) / plan.slug
        output_dir.mkdir(parents=True, exist_ok=True)
        self._init_journal(plan.slug)
        (output_dir / "01_plan.json").write_text(
            plan.model_dump_json(indent=2), encoding="utf-8"
        )

        # Live progress file (Manus todo.md pattern — recency bias anchoring)
        progress_path = output_dir / "PROGRESS.md"
        def _update_progress(phase: str, detail: str = "") -> None:
            lines = [
                f"# Research: {topic[:80]}",
                f"**Current phase:** {phase}",
                f"**Detail:** {detail}" if detail else "",
                "",
                "## Completed",
            ]
            for stage_name in stages:
                done = phase in stages and stages.index(stage_name) < stages.index(phase)
                marker = "x" if done else " "
                lines.append(f"- [{marker}] {stage_name}")
            progress_path.write_text("\n".join(lines), encoding="utf-8")
        _update_progress("plan", f"{len(plan.sub_questions)} sub-questions")

        if self.config.intervene:
            if not await self._check_intervention("plan", plan):
                return output_dir

        # State accumulated across iterative research
        research_state = None

        # Phase 1.5: Recall — check knowledge layer for prior research on this topic
        prior_knowledge = []
        if self.doc_store.count() > 0:
            self.index.build()
            prior_hits = self.index.search_hybrid(plan.topic, limit=5)
            if prior_hits:
                console.print(f"\n[bold]Phase 1.5:[/bold] Recalled {len(prior_hits)} prior knowledge chunks")
                for chunk, score in prior_hits:
                    prior_knowledge.append({
                        "title": chunk.doc_id, "content": chunk.text[:500],
                        "score": score, "_source": "knowledge_store",
                    })
                self._log(JournalEntry(
                    timestamp=self._now(), phase="recall",
                    action="prior_knowledge",
                    result_summary=f"Recalled {len(prior_hits)} relevant chunks from prior research",
                ))

        # Phase 2: Search
        _update_progress("search", "Querying arxiv, github, semantic scholar...")
        findings = []
        if self._stage_enabled("search"):
            findings = await self._search(plan)
            (output_dir / "02_sources.json").write_text(
                json.dumps([f.model_dump() for f in findings], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # Phase 2.5: Relevance filter
        _update_progress("filter", f"Scoring {sum(len(f.results) for f in findings)} results...")
        if self._stage_enabled("filter") and findings:
            findings = await self._filter_relevant(plan.topic, findings)
            (output_dir / "02_filtered.json").write_text(
                json.dumps([f.model_dump() for f in findings], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        # Phase 2.7: Deep fetch
        _update_progress("deep_fetch", f"Downloading {sum(len(f.results) for f in findings)} sources...")
        if self._stage_enabled("deep_fetch") and findings:
            findings = await self._deep_fetch(findings, output_dir)

        if self.config.intervene:
            if not await self._check_intervention("search", plan, findings=findings):
                return output_dir

        # Phase 3: Analyze
        _update_progress("analyze", f"Analyzing {sum(len(f.results) for f in findings)} sources...")
        analyses = []
        if self._stage_enabled("analyze") and findings:
            analyses = await self._analyze(findings)
            summaries_dir = output_dir / "05_summaries"
            summaries_dir.mkdir(exist_ok=True)
            for i, a in enumerate(analyses):
                safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in a.title[:50]).strip()
                (summaries_dir / f"{i:03d}_{safe_title}.json").write_text(
                    a.model_dump_json(indent=2), encoding="utf-8"
                )

        # Phase 3.5: Iterative retrieval (ITER-RETGEN pattern)
        # Perplexity-style: search → read → extract facts → refine queries → repeat
        _update_progress("iterative", f"ITER-RETGEN: deepening {len(analyses)} analyses...")
        if self._stage_enabled("iterative") and analyses:
            from core.iterative_researcher import IterativeResearcher
            console.print("\n[bold]Phase 3.5:[/bold] Iterative retrieval (ITER-RETGEN)...")

            async def _search_adapter(query: str, source: str) -> list[dict]:
                """Adapter: route search to our tool registry."""
                tool_map = {
                    "arxiv": "search_arxiv", "github": "search_github",
                    "semantic_scholar": "search_semantic_scholar",
                    "web": "search_arxiv",  # fallback
                }
                tool_name = tool_map.get(source, "search_arxiv")
                cache_key = f"{tool_name}:{query}:{self.config.max_results_per_source}"
                cached = self.cache.get("search", cache_key)
                if cached is not None:
                    return cached
                result = await self.registry.execute(
                    tool_name, query=query, max_results=self.config.max_results_per_source,
                )
                if result.success and result.data:
                    data = result.data if isinstance(result.data, list) else []
                    if data:
                        self.cache.set("search", cache_key, data, ttl_seconds=3600)
                    for item in data:
                        item["_source_type"] = source
                    return data
                return []

            # Build initial queries from first-round analyses
            initial_queries = [
                {"query": a.title, "target_source": "arxiv", "reason": "deepen analysis"}
                for a in analyses[:5]
            ]

            researcher = IterativeResearcher(
                llm=self.llm,
                search_fn=_search_adapter,
                max_iterations=3,
                novelty_threshold=0.1,
                checkpoint_path=str(output_dir / "03b_research_state.json"),
            )
            research_state = await researcher.research(plan.topic, initial_queries)

            self._log(JournalEntry(
                timestamp=self._now(), phase="search",
                action="iterative_research_done",
                result_summary=(
                    f"{len(research_state.iterations)} iterations, "
                    f"{len(research_state.fact_bank)} facts, "
                    f"{len(research_state.all_sources)} sources, "
                    f"{research_state.total_queries} queries"
                ),
            ))
            console.print(
                f"  [green]Iterative research: {len(research_state.fact_bank)} facts from "
                f"{len(research_state.all_sources)} sources ({research_state.total_queries} queries)[/green]"
            )

            # Convert new sources to findings → analyses
            extra_findings = []
            for src in research_state.all_sources:
                if src.content or src.snippet:
                    extra_findings.append(Finding(
                        source=src.source_type,
                        query=plan.topic,
                        results=[{
                            "title": src.title, "url": src.url,
                            "abstract": src.snippet, "_full_text": src.content,
                            "_source_type": src.source_type,
                        }],
                    ))

            if extra_findings:
                if self._stage_enabled("filter"):
                    extra_findings = await self._filter_relevant(plan.topic, extra_findings)
                extra_analyses = await self._analyze(extra_findings)
                analyses.extend(extra_analyses)
                summaries_dir = output_dir / "05_summaries"
                summaries_dir.mkdir(exist_ok=True)
                for i, a in enumerate(extra_analyses, len(analyses) - len(extra_analyses)):
                    safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in a.title[:50]).strip()
                    (summaries_dir / f"{i:03d}_{safe_title}.json").write_text(
                        a.model_dump_json(indent=2), encoding="utf-8"
                    )

            # Save fact bank and research state
            (output_dir / "03b_fact_bank.json").write_text(
                json.dumps([{
                    "claim": f.claim, "source": f.source_title,
                    "confidence": f.confidence, "corroboration": f.corroboration_count,
                } for f in research_state.fact_bank], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # Phase 3.7: Hypothesis generation
        _update_progress("hypotheses", f"Generating from {len(analyses)} analyses + {len(research_state.fact_bank) if research_state else 0} facts...")
        hypotheses = {}
        if self._stage_enabled("hypotheses") and analyses:
            hypotheses = await self._generate_hypotheses(plan, analyses, research_state=research_state)
            (output_dir / "04_hypotheses.json").write_text(
                json.dumps(hypotheses, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        # Phase 3.8: Contradiction detection
        _update_progress("contradictions", "Cross-checking claims...")
        if self._stage_enabled("contradictions") and analyses:
            console.print("\n[bold]Phase 3.8:[/bold] Detecting contradictions...")
            self._log(JournalEntry(
                timestamp=self._now(), phase="analyze",
                action="contradiction_detection", result_summary="Checking cross-source contradictions",
            ))
            try:
                source_dicts = [a.model_dump(exclude={"raw_source"}) for a in analyses[:15]]
                contradictions = await self.claim_verifier.detect_contradictions(source_dicts)
                c_count = len(contradictions.get("contradictions", []))
                consensus_count = len(contradictions.get("consensus", []))
                console.print(f"  Found {c_count} contradictions, {consensus_count} consensus points")
                self._log(JournalEntry(
                    timestamp=self._now(), phase="analyze",
                    action="contradictions_found",
                    result_summary=f"{c_count} contradictions, {consensus_count} consensus",
                ))
                hypotheses["contradictions"] = contradictions.get("contradictions", [])
                hypotheses["consensus"] = contradictions.get("consensus", [])
                if hypotheses:
                    (output_dir / "04_hypotheses.json").write_text(
                        json.dumps(hypotheses, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
            except Exception as e:
                logger.warning("Contradiction detection failed: %s", e)

        # Phase 4: Synthesize
        _update_progress("synthesize", f"Writing report from {len(analyses)} analyses, {len(hypotheses.get('hypotheses', []))} hypotheses...")
        report_path = await self._synthesize(plan, analyses, output_dir, hypotheses, research_state=research_state)

        console.print(Panel(
            f"[bold green]Research complete![/bold green]\n"
            f"Output: {output_dir}\n"
            f"Report: {report_path}",
            style="green",
        ))
        return output_dir

    # ------------------------------------------------------------------
    # Phase 1: Planning
    # ------------------------------------------------------------------

    async def _plan(self, topic: str) -> ResearchPlan:
        console.print("[bold]Phase 1:[/bold] Generating research plan...")
        self._log(JournalEntry(
            timestamp=self._now(), phase="plan",
            action="generate_plan", result_summary=f"Topic: {topic}",
        ))

        plan = await self.planner.generate_plan(topic)

        for sq in plan.sub_questions:
            console.print(f"  [dim]P{sq.priority}[/dim] {sq.question}")

        self._log(JournalEntry(
            timestamp=self._now(), phase="plan",
            action="plan_generated",
            result_summary=f"{len(plan.sub_questions)} sub-questions",
        ))
        return plan

    # ------------------------------------------------------------------
    # Phase 2: Search
    # ------------------------------------------------------------------

    async def _search(self, plan: ResearchPlan) -> list[Finding]:
        console.print("\n[bold]Phase 2:[/bold] Searching sources...")
        # Invalidate stale caches (>24h) before searching
        stale_count = self.refresh_scheduler.invalidate_stale(max_age_hours=24)
        if stale_count:
            console.print(f"  [dim]Invalidated {stale_count} stale source caches[/dim]")
        findings: list[Finding] = []

        tasks = []
        for sq in plan.sub_questions:
            for source in sq.sources:
                if source not in self.config.sources:
                    continue
                tool_name = f"search_{source}"
                if not self.registry.get(tool_name):
                    continue
                tasks.append((sq, source, tool_name))

        async def _do_search(sq: SubQuestion, source: str, tool_name: str) -> Finding | None:
            # Use keywords for API search (always English), fall back to question
            if sq.keywords:
                query = " ".join(sq.keywords)
            else:
                query = sq.question
            # If query contains non-ASCII (e.g. Cyrillic), prefer keywords
            if any(ord(c) > 127 for c in query) and sq.keywords:
                query = " ".join(sq.keywords)
            cache_key = f"{tool_name}:{query}:{self.config.max_results_per_source}"

            # Check cache first
            cached = self.cache.get("search", cache_key)
            if cached is not None:
                console.print(f"  [dim]{tool_name}:[/dim] {query[:60]}... [green](cached)[/green]")
                self._log(JournalEntry(
                    timestamp=self._now(), phase="search",
                    action="cache_hit", tool_name=tool_name,
                    result_summary=f"{len(cached)} cached results",
                ))
                return Finding(source=source, query=query, results=cached)

            console.print(f"  [dim]{tool_name}:[/dim] {query[:60]}...")

            self._log(JournalEntry(
                timestamp=self._now(), phase="search",
                action="tool_call", tool_name=tool_name,
                tool_input={"query": query, "max_results": self.config.max_results_per_source},
            ))

            result = await self.registry.execute(
                tool_name,
                query=query,
                max_results=self.config.max_results_per_source,
            )

            if result.success and result.data:
                data = result.data if isinstance(result.data, list) else []
                # Cache only non-empty results for 1 hour
                if data:
                    self.cache.set("search", cache_key, data, ttl_seconds=3600)
                # Register sources
                for item in data:
                    url = item.get("url") or item.get("pdf_url") or item.get("paper_url") or ""
                    if url:
                        self.source_registry.register(url, source)
                self._log(JournalEntry(
                    timestamp=self._now(), phase="search",
                    action="tool_result", tool_name=tool_name,
                    result_summary=f"{len(data)} results",
                ))
                return Finding(source=source, query=query, results=data)
            else:
                self._log(JournalEntry(
                    timestamp=self._now(), phase="search",
                    action="tool_error", tool_name=tool_name,
                    result_summary=result.error or "no results",
                ))
                return None

        if self.config.parallel_search:
            results = await asyncio.gather(
                *[_do_search(sq, src, tn) for sq, src, tn in tasks],
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, Finding):
                    findings.append(r)
                elif isinstance(r, Exception):
                    logger.error("Search task failed: %s", r)
        else:
            for sq, src, tn in tasks:
                r = await _do_search(sq, src, tn)
                if r:
                    findings.append(r)

        # Deduplicate by URL across all findings
        seen_urls: set[str] = set()
        total_before = sum(len(f.results) for f in findings)
        for finding in findings:
            deduped = []
            for item in finding.results:
                url = item.get("url") or item.get("pdf_url") or item.get("paper_url") or ""
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                deduped.append(item)
            finding.results = deduped
        total_after = sum(len(f.results) for f in findings)
        dupes = total_before - total_after

        console.print(f"  [green]Found {total_after} unique results ({dupes} duplicates removed)[/green]")
        return findings

    # ------------------------------------------------------------------
    # Phase 2.5: Relevance filter
    # ------------------------------------------------------------------

    async def _filter_relevant(self, topic: str, findings: list[Finding]) -> list[Finding]:
        console.print("\n[bold]Phase 2.5:[/bold] Filtering by relevance (parallel)...")

        # Collect all items to score
        all_items: list[tuple[Finding, dict]] = []
        for finding in findings:
            for item in finding.results:
                if item.get("title") or item.get("name"):
                    all_items.append((finding, item))

        # Score items in parallel (4 concurrent to avoid overwhelming Ollama)
        sem = asyncio.Semaphore(4)
        scored: list[tuple[Finding, dict, int, str]] = []

        async def _score_one(finding: Finding, item: dict) -> tuple[Finding, dict, int, str]:
            title = item.get("title") or item.get("name") or ""
            abstract = item.get("abstract") or item.get("description") or ""
            prompt = self._inject_date(
                RELEVANCE_FILTER
                .replace("{{ topic }}", topic)
                .replace("{{ title }}", title)
                .replace("{{ abstract }}", abstract[:500])
            )
            try:
                async with sem:
                    raw = await self.llm.generate(prompt, mode=LLMMode.FAST)
                data = self.planner._extract_json(raw)
                score = int(data.get("score", 0))
                reason = data.get("reason", "")
                self._log(JournalEntry(
                    timestamp=self._now(), phase="search",
                    action="relevance_check",
                    result_summary=f"score={score}/10 {title[:50]} | {reason[:50]}",
                ))
                return (finding, item, score, reason)
            except Exception:
                return (finding, item, 5, "error-kept")

        results = await asyncio.gather(
            *[_score_one(f, i) for f, i in all_items],
            return_exceptions=True,
        )

        # Build filtered findings
        finding_items: dict[str, list[dict]] = {}
        for r in results:
            if isinstance(r, Exception):
                continue
            finding, item, score, reason = r
            title = item.get("title") or item.get("name") or ""
            if score >= 5:
                item["_relevance_score"] = score
                item["_relevance_reason"] = reason
                key = f"{finding.source}:{finding.query}"
                finding_items.setdefault(key, []).append(item)
            else:
                console.print(f"  [dim]Filtered out (score={score}):[/dim] {title[:60]}")

        filtered_findings = []
        for finding in findings:
            key = f"{finding.source}:{finding.query}"
            items = finding_items.get(key, [])
            if items:
                filtered_findings.append(Finding(
                    source=finding.source, query=finding.query, results=items,
                ))

        total_before = sum(len(f.results) for f in findings)
        total_after = sum(len(f.results) for f in filtered_findings)
        console.print(f"  [green]Kept {total_after}/{total_before} relevant results[/green]")
        return filtered_findings

    # ------------------------------------------------------------------
    # Phase 2.7: Deep fetch
    # ------------------------------------------------------------------

    async def _deep_fetch(self, findings: list[Finding], output_dir: Path) -> list[Finding]:
        console.print("\n[bold]Phase 2.7:[/bold] Deep fetching top sources (parallel)...")
        papers_dir = output_dir / "03_papers"
        papers_dir.mkdir(exist_ok=True)
        code_dir = output_dir / "04_code"
        code_dir.mkdir(exist_ok=True)

        async def _fetch_one(item: dict) -> None:
            """Fetch full text for a single item (PDF or GitHub README)."""
            # Fetch PDF for papers
            pdf_url = item.get("pdf_url")
            if pdf_url:
                try:
                    safe_name = "".join(c if c.isalnum() or c in "-_" else "_"
                                       for c in (item.get("title", "paper"))[:40])
                    save_path = str(papers_dir / f"{safe_name}.pdf")
                    result = await self.registry.execute(
                        "download_pdf", url=pdf_url, save_path=save_path
                    )
                    if result.success:
                        parse_result = await self.registry.execute(
                            "parse_pdf", file_path=save_path, max_chars=6000
                        )
                        if parse_result.success and parse_result.data:
                            item["_full_text"] = parse_result.data.get("text", "")
                            self._log(JournalEntry(
                                timestamp=self._now(), phase="search",
                                action="deep_fetch_pdf",
                                result_summary=f"Parsed {parse_result.data.get('num_pages', 0)} pages: {item.get('title', '')[:50]}",
                            ))
                except Exception as e:
                    logger.debug("PDF fetch failed for %s: %s", pdf_url, e)

            # Fetch README for GitHub repos
            repo_url = item.get("url", "")
            if "github.com" in repo_url and not item.get("_full_text"):
                try:
                    result = await self.registry.execute("inspect_code", url=repo_url)
                    if result.success and result.data:
                        item["_full_text"] = result.data.get("content", "")
                        self._log(JournalEntry(
                            timestamp=self._now(), phase="search",
                            action="deep_fetch_readme",
                            result_summary=f"Fetched README: {item.get('name', item.get('title', ''))[:50]}",
                        ))
                except Exception as e:
                    logger.debug("README fetch failed for %s: %s", repo_url, e)

        # Collect all items to fetch and run in parallel (max 8 concurrent)
        all_items = [item for f in findings for item in f.results[:10]]
        sem = asyncio.Semaphore(8)

        async def _limited_fetch(item: dict) -> None:
            async with sem:
                await _fetch_one(item)

        await asyncio.gather(*[_limited_fetch(item) for item in all_items], return_exceptions=True)

        fetched = sum(1 for f in findings for i in f.results if i.get("_full_text"))
        console.print(f"  [green]Deep fetched {fetched} full texts[/green]")
        return findings

    # ------------------------------------------------------------------
    # Phase 3.5: Search refined queries
    # ------------------------------------------------------------------

    async def _search_refined(self, refined_queries) -> list[Finding]:
        """Execute refined search queries from iterative deepening."""
        findings: list[Finding] = []
        for rq in refined_queries:
            tool_name = f"search_{rq.source}"
            if not self.registry.get(tool_name):
                continue

            self._log(JournalEntry(
                timestamp=self._now(), phase="search",
                action="refined_search", tool_name=tool_name,
                tool_input={"query": rq.query},
                result_summary=f"Reason: {rq.reason[:60]}",
            ))

            result = await self.registry.execute(tool_name, query=rq.query, max_results=5)
            if result.success and result.data:
                data = result.data if isinstance(result.data, list) else []
                if data:
                    findings.append(Finding(source=rq.source, query=rq.query, results=data))

        return findings

    # ------------------------------------------------------------------
    # Phase 3.7: Hypothesis generation
    # ------------------------------------------------------------------

    async def _generate_hypotheses(
        self, plan: ResearchPlan, analyses: list[AnalysisResult],
        research_state: Any | None = None,
    ) -> dict:
        console.print("\n[bold]Phase 3.7:[/bold] Generating hypotheses...")

        analyses_summary = "\n".join(
            f"- {a.title}: {a.approach[:150]}"
            f"\n  Strengths: {', '.join(a.strengths[:3])}"
            f"\n  Weaknesses: {', '.join(a.weaknesses[:3])}"
            for a in analyses[:15]
        )

        # Enrich with verified facts from iterative research
        facts_context = ""
        if research_state and research_state.fact_bank:
            facts_context = (
                "\n\nVerified facts from iterative research (use these to ground hypotheses):\n"
                + research_state.get_fact_summary(20)
            )

        prompt = self._inject_date(
            HYPOTHESIS_GENERATION
            .replace("{{ topic }}", plan.topic)
            .replace("{{ analyses_summary }}", analyses_summary[:7000] + facts_context)
        )

        raw = await self.llm.generate(prompt, mode=LLMMode.THINKING)
        data = self.planner._extract_json(raw)

        hypotheses = data.get("hypotheses", [])
        gaps = data.get("gaps", [])
        uncertainties = data.get("uncertainties", [])

        for h in hypotheses:
            console.print(f"  [cyan]H{h.get('id', '?')}:[/cyan] {h.get('title', '')}")

        self._log(JournalEntry(
            timestamp=self._now(), phase="analyze",
            action="hypotheses_generated",
            result_summary=f"{len(hypotheses)} hypotheses, {len(gaps)} gaps, {len(uncertainties)} uncertainties",
        ))

        return {"hypotheses": hypotheses, "gaps": gaps, "uncertainties": uncertainties}

    # ------------------------------------------------------------------
    # Phase 3: Analysis
    # ------------------------------------------------------------------

    async def _analyze(self, findings: list[Finding]) -> list[AnalysisResult]:
        console.print("\n[bold]Phase 3:[/bold] Analyzing findings...")
        analyses: list[AnalysisResult] = []

        # Deduplicate by title
        seen_titles: set[str] = set()
        unique_items: list[tuple[str, dict]] = []

        for finding in findings:
            for item in finding.results:
                title = item.get("title") or item.get("name") or ""
                title_lower = title.lower().strip()
                if title_lower and title_lower not in seen_titles:
                    seen_titles.add(title_lower)
                    unique_items.append((finding.source, item))

        console.print(f"  Analyzing {len(unique_items)} unique sources...")

        for source_type, item in unique_items:
            full_text = item.get("_full_text", "")
            title = item.get("title") or item.get("name") or ""
            url = item.get("url") or item.get("pdf_url") or ""

            # Check analysis cache (24h TTL) — avoid re-analyzing same paper
            cache_key = f"analysis:{url or title}"
            cached_analysis = self.cache.get("analysis", cache_key)
            if cached_analysis is not None:
                try:
                    analysis = AnalysisResult(raw_source=item, **{
                        k: v for k, v in cached_analysis.items()
                        if k in AnalysisResult.model_fields
                    })
                    analyses.append(analysis)
                    console.print(f"  [dim]{title[:60]}[/dim] [green](cached)[/green]")
                    continue
                except Exception:
                    pass  # cache corrupted, re-analyze

            if full_text:
                prompt = self._inject_date(
                    DEEP_ANALYSIS
                    .replace("{{ source_type }}", source_type)
                    .replace("{{ title }}", title)
                    .replace("{{ content }}", full_text[:6000])
                )
                action = "deep_analyze_source"
            else:
                content = json.dumps(item, ensure_ascii=False, default=str)
                prompt = self._inject_date(
                    ANALYSIS
                    .replace("{{ source_type }}", source_type)
                    .replace("{{ content }}", content[:4000])
                )
                action = "analyze_source"

            self._log(JournalEntry(
                timestamp=self._now(), phase="analyze",
                action=action,
                result_summary=f"{'[DEEP] ' if full_text else ''}{title[:70]}",
            ))

            try:
                raw = await self.llm.generate(prompt, mode=LLMMode.FAST)
                data = self.planner._extract_json(raw)
                analysis = AnalysisResult(raw_source=item, **{
                    k: v for k, v in data.items()
                    if k in AnalysisResult.model_fields
                })
                if not analysis.title:
                    analysis.title = item.get("title", item.get("name", "unknown"))
                analyses.append(analysis)
                # Cache analysis result (24h)
                self.cache.set("analysis", cache_key, data, ttl_seconds=86400)
            except Exception as e:
                logger.warning("Failed to analyze item: %s", e)
                analyses.append(AnalysisResult(
                    title=item.get("title", item.get("name", "unknown")),
                    approach=item.get("abstract", item.get("description", "")),
                    raw_source=item,
                ))

        console.print(f"  [green]Analyzed {len(analyses)} sources[/green]")
        return analyses

    # ------------------------------------------------------------------
    # Phase 4: Synthesis
    # ------------------------------------------------------------------

    async def _synthesize(
        self,
        plan: ResearchPlan,
        analyses: list[AnalysisResult],
        output_dir: Path,
        hypotheses: dict | None = None,
        research_state: Any | None = None,
    ) -> Path:
        console.print("\n[bold]Phase 4:[/bold] Synthesizing report...")

        self._log(JournalEntry(
            timestamp=self._now(), phase="synthesize",
            action="start_synthesis",
            result_summary=f"{len(analyses)} analyses to synthesize",
        ))

        # Build analyses summary for prompt
        analyses_text = "\n\n".join(
            f"### {a.title}\n"
            f"Approach: {a.approach}\n"
            f"Strengths: {', '.join(a.strengths)}\n"
            f"Weaknesses: {', '.join(a.weaknesses)}\n"
            f"Code: {a.relevant_code}"
            for a in analyses
        )

        # Inject verified facts and pre-embedded citations (Perplexity-style)
        facts_section = ""
        citations_section = ""
        if research_state:
            fact_summary = research_state.get_fact_summary(30)
            if fact_summary and fact_summary != "(no facts yet)":
                facts_section = f"\n\n## Verified Facts (from iterative research)\n{fact_summary}"
            citation_block = research_state.get_citation_block(8)
            if citation_block:
                citations_section = f"\n\n## Source Citations (use these in your report)\n{citation_block}"

        plan_summary = "\n".join(
            f"- {sq.question}" for sq in plan.sub_questions
        )

        # Add hypotheses to prompt if available
        hypotheses_text = ""
        if hypotheses and hypotheses.get("hypotheses"):
            h_lines = []
            for h in hypotheses["hypotheses"]:
                h_lines.append(f"- **{h.get('title', '')}**: {h.get('description', '')}")
                h_lines.append(f"  Validation: {h.get('validation_method', 'N/A')}")
            hypotheses_text = "\n\nHypotheses for implementation:\n" + "\n".join(h_lines)

            gaps = hypotheses.get("gaps", [])
            if gaps:
                hypotheses_text += "\n\nKnowledge gaps:\n" + "\n".join(f"- {g}" for g in gaps)

            uncertainties = hypotheses.get("uncertainties", [])
            if uncertainties:
                hypotheses_text += "\n\nUncertainties:\n" + "\n".join(f"- {u}" for u in uncertainties)

        # Build numbered reference map from REAL sources (not LLM-generated)
        ref_map: dict[int, dict] = {}
        for i, a in enumerate(analyses, 1):
            src = a.raw_source
            url = src.get("url") or src.get("pdf_url") or src.get("paper_url") or src.get("html_url", "")
            ref_map[i] = {"title": a.title, "url": url}

        reference_list = "\n".join(
            f"[{i}] {r['title']} — {r['url']}" for i, r in ref_map.items()
        )

        # Inject reference map into prompt so LLM uses real [N] numbers
        ref_instruction = (
            f"\n\nAVAILABLE SOURCES (use ONLY these [N] citations, do NOT invent URLs):\n"
            f"{reference_list}\n\n"
            f"RULES: Use [1], [2], etc. to cite. The References section is already built — "
            f"do NOT write your own References section. Do NOT generate URLs."
        )

        prompt = self._inject_date(
            SYNTHESIS
            .replace("{{ topic }}", plan.topic)
            .replace("{{ plan_summary }}", plan_summary)
            .replace("{{ analyses }}", analyses_text[:7000] + ref_instruction + facts_section + citations_section + hypotheses_text)
        )

        report = await self.llm.generate(prompt, mode=LLMMode.THINKING)

        # Post-process: strip any LLM-generated References section (we build our own)
        import re
        report = re.split(r'\n## (?:8\.|9\.)?\s*References\b', report, maxsplit=1)[0]

        # Append real references
        report += "\n\n## References\n\n" + "\n".join(
            f"[{i}] **{r['title']}** — {r['url']}" for i, r in ref_map.items()
        ) + "\n"

        # Validate: flag fake URLs and suspicious metrics
        fake_urls = re.findall(r'https?://example\.com\S*', report)
        if fake_urls:
            report = re.sub(r'https?://example\.com\S*', '[URL not available]', report)
            report += f"\n\n> **WARNING**: {len(fake_urls)} placeholder URL(s) replaced with [URL not available].\n"

        # Flag suspicious metrics (LLM-hallucinated numbers without source)
        suspicious = re.findall(r'(?:achieves?|scores?|reaches?|obtains?)\s+(\d{2,3}%|\d\.\d+)', report)
        # Only flag if the number doesn't appear near a citation [N]
        unfounded_metrics = []
        for match in suspicious:
            # Check if there's a [N] within 50 chars
            idx = report.find(match)
            if idx >= 0:
                context = report[max(0, idx-20):idx+len(match)+30]
                if not re.search(r'\[\d+\]', context):
                    unfounded_metrics.append(match)
        if unfounded_metrics:
            report += f"\n> **NOTE**: {len(unfounded_metrics)} metric(s) without citation — may be LLM-generated: {', '.join(unfounded_metrics[:5])}\n"

        # Save synthesis
        report_path = output_dir / "07_synthesis.md"
        report_path.write_text(report, encoding="utf-8")

        # Build comparison table
        methods_json = json.dumps(
            [{"title": a.title, "approach": a.approach, "strengths": a.strengths,
              "weaknesses": a.weaknesses, "code": a.relevant_code}
             for a in analyses],
            ensure_ascii=False,
        )
        comparison_result = await self.registry.execute(
            "compare_methods", methods_json=methods_json
        )
        comparison_md = ""
        if comparison_result.success and comparison_result.data:
            comparison_md = comparison_result.data.get("table", "")

        (output_dir / "06_comparison.md").write_text(comparison_md, encoding="utf-8")

        # Build references
        refs: list[str] = []
        for i, a in enumerate(analyses, 1):
            src = a.raw_source
            url = src.get("url") or src.get("pdf_url") or src.get("paper_url") or src.get("html_url", "")
            refs.append(f"{i}. **{a.title}** — {url}")

        (output_dir / "08_references.md").write_text(
            "# References\n\n" + "\n".join(refs), encoding="utf-8"
        )

        # Store documents in knowledge layer + build search index
        for a in analyses:
            self.doc_store.add(Document(
                doc_id=a.title[:60],
                source_url=a.raw_source.get("url", ""),
                title=a.title,
                content=a.approach + " " + " ".join(a.strengths + a.weaknesses),
                doc_type="analysis",
            ))
        # Build TF-IDF index + persist knowledge to disk
        if self.doc_store.count() > 0:
            self.index.build()
            self.doc_store.save(self._knowledge_dir / "documents.json")
            self.source_registry.save(self._knowledge_dir / "sources.json")

        # Evaluate report quality
        source_dicts = []
        for f in []:  # findings not in scope here, use analyses
            pass
        for a in analyses:
            source_dicts.append({"source": "analysis", "title": a.title})

        plan_questions = [sq.question for sq in plan.sub_questions]
        metrics = self.evaluator.evaluate_report(report, source_dicts, plan_questions)

        eval_data = {
            "groundedness": round(metrics.groundedness, 3),
            "coverage": round(metrics.coverage, 3),
            "source_diversity": round(metrics.source_diversity, 3),
            "code_presence": round(metrics.code_presence, 3),
            "overall": round(metrics.overall, 3),
            "sources_registered": self.source_registry.count(),
            "documents_stored": self.doc_store.count(),
            "chunks_indexed": self.doc_store.total_chunks(),
        }

        (output_dir / "09_evaluation.json").write_text(
            json.dumps(eval_data, indent=2), encoding="utf-8"
        )

        console.print(f"  [cyan]Quality score: {metrics.overall:.1%}[/cyan]")

        self._log(JournalEntry(
            timestamp=self._now(), phase="synthesize",
            action="evaluation_complete",
            result_summary=f"overall={metrics.overall:.3f} coverage={metrics.coverage:.3f}",
        ))

        self._log(JournalEntry(
            timestamp=self._now(), phase="synthesize",
            action="synthesis_complete",
            result_summary=f"Report: {report_path}",
        ))

        return report_path

    # ------------------------------------------------------------------
    # ReAct loop (for future use with more complex tasks)
    # ------------------------------------------------------------------

    async def _react_loop(self, task: str, max_steps: int = 15) -> str:
        system = SYSTEM_REACT.replace(
            "{{ tool_definitions }}", self.registry.format_for_prompt()
        )

        history = f"Task: {task}\n\n"

        for step in range(max_steps):
            response = await self.llm.generate(history, mode=LLMMode.THINKING, system=system)
            history += response + "\n"

            action_name, action_input, final_answer = _parse_react(response)

            if final_answer:
                return final_answer

            if action_name:
                self._log(JournalEntry(
                    timestamp=self._now(), phase="search",
                    action="react_tool_call", tool_name=action_name,
                    tool_input=action_input,
                ))

                result = await self.registry.execute(action_name, **(action_input or {}))
                observation = result.to_observation()
                history += f"\nObservation: {observation}\n\n"

                self._log(JournalEntry(
                    timestamp=self._now(), phase="search",
                    action="react_observation", tool_name=action_name,
                    result_summary=observation[:200],
                ))
            else:
                # Model didn't produce a valid action or final answer
                history += "\nYour response did not follow the required format. Please respond with either an Action or a Final Answer.\n\n"

        return "Max steps reached without a final answer."

    # ------------------------------------------------------------------
    # Intervention
    # ------------------------------------------------------------------

    async def _check_intervention(
        self,
        phase: str,
        plan: ResearchPlan,
        findings: list[Finding] | None = None,
    ) -> bool:
        summary = f"Phase: {phase}\nTopic: {plan.topic}\n"
        if findings:
            summary += f"Findings: {sum(len(f.results) for f in findings)} results from {len(findings)} queries\n"

        console.print(Panel(summary, title="[yellow]Intervention Point[/yellow]", style="yellow"))
        response = await asyncio.to_thread(
            input, "Press Enter to continue, or type 'abort' to stop: "
        )
        return response.strip().lower() != "abort"

    # ------------------------------------------------------------------
    # Journal
    # ------------------------------------------------------------------

    def _init_journal(self, slug: str) -> None:
        journal_dir = Path(self.config.journal_dir)
        journal_dir.mkdir(parents=True, exist_ok=True)
        self._journal_path = journal_dir / f"{slug}.jsonl"

    def _log(self, entry: JournalEntry) -> None:
        if self._journal_path:
            with open(self._journal_path, "a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")

        if self.config.verbose:
            console.print(
                f"  [dim][{entry.phase}] {entry.action}[/dim]"
                + (f" -> {entry.result_summary[:80]}" if entry.result_summary else "")
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _inject_date(text: str) -> str:
        """Replace date placeholders in prompts."""
        now = datetime.now(timezone.utc)
        return (
            text
            .replace("{{ current_date }}", now.strftime("%Y-%m-%d"))
            .replace("{{ current_year }}", str(now.year))
        )
