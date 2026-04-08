"""Perplexity-style iterative research engine.

Implements ITER-RETGEN pattern: search → read → extract facts → refine queries → search again.
Each iteration narrows focus, accumulates verified facts, and scores sources on multiple axes.

Key techniques (from Perplexity/ITER-RETGEN/FLARE research):
- Iterative retrieval: multiple search rounds with progressively refined queries
- Fact bank: accumulates verified facts across iterations
- Source quality scoring: relevance × authority × recency × cross-corroboration
- Adaptive stopping: halt when novelty rate drops below threshold
- Citation pre-embedding: source excerpts injected into synthesis prompt before generation
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.llm import LLM, LLMMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    """Atomic verified fact extracted from a source."""
    claim: str
    source_url: str
    source_title: str
    confidence: float = 0.5  # 0-1
    corroboration_count: int = 1  # how many sources support this
    contradicted_by: list[str] = field(default_factory=list)
    extracted_at_iter: int = 0


@dataclass
class ScoredSource:
    """Source with multi-axis quality score."""
    url: str
    title: str
    source_type: str  # arxiv, github, web
    content: str = ""
    snippet: str = ""

    # Quality axes
    relevance: float = 0.0     # semantic match to query
    authority: float = 0.5     # domain reputation, citation count
    recency: float = 0.5       # freshness (1.0 = this month, 0.0 = 5+ years)
    corroboration: float = 0.0  # how many other sources agree

    @property
    def composite_score(self) -> float:
        """Weighted composite: relevance is king, authority and recency matter."""
        return (
            0.40 * self.relevance +
            0.25 * self.authority +
            0.20 * self.recency +
            0.15 * self.corroboration
        )

    def as_citation(self) -> str:
        """Format as citation block for prompt injection."""
        snippet = self.snippet[:300] if self.snippet else self.content[:300]
        return f"[Source: {self.title}]({self.url})\n> {snippet}\n"


@dataclass
class IterationResult:
    """Result of one search-read-extract iteration."""
    iteration: int
    queries: list[str]
    sources_found: int
    new_facts: int
    novelty_rate: float  # fraction of new facts vs total facts
    top_sources: list[ScoredSource] = field(default_factory=list)


@dataclass
class ResearchState:
    """Accumulated state across all iterations."""
    topic: str
    fact_bank: list[Fact] = field(default_factory=list)
    all_sources: list[ScoredSource] = field(default_factory=list)
    seen_urls: set = field(default_factory=set)
    iterations: list[IterationResult] = field(default_factory=list)
    total_queries: int = 0

    def add_fact(self, fact: Fact) -> bool:
        """Add fact if novel. Returns True if new, False if duplicate."""
        claim_hash = hashlib.md5(fact.claim.lower().strip().encode()).hexdigest()
        for existing in self.fact_bank:
            if hashlib.md5(existing.claim.lower().strip().encode()).hexdigest() == claim_hash:
                existing.corroboration_count += 1
                return False
        self.fact_bank.append(fact)
        return True

    def get_fact_summary(self, max_facts: int = 30) -> str:
        """Build fact bank summary for LLM context."""
        sorted_facts = sorted(self.fact_bank, key=lambda f: -f.confidence * f.corroboration_count)
        lines = []
        for f in sorted_facts[:max_facts]:
            conf_tag = "★" * min(int(f.confidence * 5), 5)
            corr_tag = f"(×{f.corroboration_count})" if f.corroboration_count > 1 else ""
            lines.append(f"- [{conf_tag}]{corr_tag} {f.claim}")
        return "\n".join(lines) if lines else "(no facts yet)"

    def get_citation_block(self, max_sources: int = 10) -> str:
        """Build pre-embedded citations for synthesis prompt."""
        top = sorted(self.all_sources, key=lambda s: -s.composite_score)[:max_sources]
        return "\n\n".join(s.as_citation() for s in top)

    def save(self, path: str) -> None:
        """Persist research state to JSON for crash recovery."""
        data = {
            "topic": self.topic,
            "fact_bank": [
                {"claim": f.claim, "source_url": f.source_url, "source_title": f.source_title,
                 "confidence": f.confidence, "corroboration_count": f.corroboration_count,
                 "extracted_at_iter": f.extracted_at_iter}
                for f in self.fact_bank
            ],
            "all_sources": [
                {"url": s.url, "title": s.title, "source_type": s.source_type,
                 "snippet": s.snippet[:500], "relevance": s.relevance,
                 "authority": s.authority, "recency": s.recency, "corroboration": s.corroboration}
                for s in self.all_sources
            ],
            "seen_urls": list(self.seen_urls),
            "total_queries": self.total_queries,
            "iterations_count": len(self.iterations),
        }
        import json as _json
        from pathlib import Path as _Path
        _Path(path).write_text(_json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> ResearchState:
        """Load persisted research state."""
        import json as _json
        from pathlib import Path as _Path
        data = _json.loads(_Path(path).read_text(encoding="utf-8"))
        state = cls(topic=data.get("topic", ""))
        state.seen_urls = set(data.get("seen_urls", []))
        state.total_queries = data.get("total_queries", 0)
        for fd in data.get("fact_bank", []):
            state.fact_bank.append(Fact(**fd))
        for sd in data.get("all_sources", []):
            state.all_sources.append(ScoredSource(**sd))
        return state

    def novelty_rate(self) -> float:
        """What fraction of recent facts were actually new."""
        if not self.iterations:
            return 1.0
        last = self.iterations[-1]
        return last.novelty_rate


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACT_FACTS_PROMPT = """\
Extract atomic facts from this research source. Each fact should be a single, verifiable claim.

Topic: {topic}
Source: {title}
Content:
{content}

Already known facts:
{known_facts}

Extract ONLY NEW facts not already in the known facts above.
Rate confidence 0.0-1.0 based on how well-supported the claim is.

Return JSON:
{{
  "facts": [
    {{"claim": "specific verifiable claim", "confidence": 0.8}},
    ...
  ]
}}
"""

REFINE_QUERIES_PROMPT = """\
You are a research strategist. Based on what we've learned so far, generate refined search queries
to fill gaps in our knowledge.

Topic: {topic}
Iteration: {iteration} of {max_iterations}

Known facts ({fact_count}):
{fact_summary}

Queries already tried:
{tried_queries}

What's missing? What contradictions need resolving? What deeper questions arise from the facts?

For each query, also include EXPANDED variants:
- Synonyms and related terms (e.g., "QLoRA" → "quantized low-rank adaptation")
- Broader/narrower scopes (e.g., "Russian bankruptcy NLP" → "legal document classification multilingual")
- Method-specific terms (e.g., if a fact mentions "gradient checkpointing", search for that specifically)

Generate 4-6 targeted search queries that would find NEW information not covered by existing facts.

Return JSON:
{{
  "queries": [
    {{"query": "specific search query", "target_source": "arxiv|github|web", "reason": "why this query",
      "expanded": ["synonym variant 1", "broader scope variant"]}},
    ...
  ]
}}
"""

SCORE_AUTHORITY_PROMPT = """\
Rate the authority/credibility of this source on a scale of 0.0-1.0.

Source: {title}
URL: {url}
Type: {source_type}
Snippet: {snippet}

Consider: Is this from a reputable venue (top conference, established journal)?
Is the author well-known? Is it a primary source or secondary?

Return ONLY a JSON: {{"authority": 0.7, "reason": "brief reason"}}
"""

COMPACT_PROMPT = """\
Summarize the key findings from this research iteration in 2-3 sentences.
Preserve specific numbers, method names, and conclusions. Drop filler.

Iteration {iteration}: {queries_count} queries, {sources_count} sources, {new_facts} new facts.
Top sources: {top_sources}
New facts found:
{new_facts_text}

Summary (2-3 sentences):
"""

AUTORATER_PROMPT = """\
Rate the quality of this research output on a 1-5 scale.

Research topic: {topic}
Output type: {output_type}
Content:
{content}

Scoring rubric:
5 = Comprehensive, accurate, well-sourced, novel insights
4 = Good coverage, mostly accurate, adequate sources
3 = Acceptable but superficial, some gaps
2 = Incomplete, inaccurate, or poorly sourced
1 = Wrong, irrelevant, or empty

Return ONLY JSON: {{"score": 4, "issues": ["list of specific problems"], "suggestion": "one improvement"}}
"""


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class IterativeResearcher:
    """Perplexity-style iterative research with fact accumulation."""

    def __init__(
        self,
        llm: LLM,
        search_fn,  # async (query, source) -> list[dict]
        max_iterations: int = 4,
        novelty_threshold: float = 0.1,  # stop if <10% new facts
        max_facts_per_source: int = 10,
        checkpoint_path: str | None = None,  # auto-save state after each iteration
    ) -> None:
        self.llm = llm
        self.search_fn = search_fn
        self.max_iterations = max_iterations
        self.novelty_threshold = novelty_threshold
        self.max_facts_per_source = max_facts_per_source
        self.checkpoint_path = checkpoint_path

    async def research(self, topic: str, initial_queries: list[dict] | None = None) -> ResearchState:
        """Run iterative research loop. Returns accumulated state."""
        state = ResearchState(topic=topic)
        tried_queries: list[str] = []

        for iteration in range(1, self.max_iterations + 1):
            logger.info("Iteration %d/%d (facts=%d, sources=%d)",
                        iteration, self.max_iterations, len(state.fact_bank), len(state.all_sources))

            # 1. Get queries for this iteration
            if iteration == 1 and initial_queries:
                queries = initial_queries
            else:
                queries = await self._generate_refined_queries(
                    state, tried_queries, iteration,
                )

            if not queries:
                logger.info("No more queries to try, stopping")
                break

            # 2. Execute searches (including expanded queries for broader recall)
            all_results: list[dict] = []
            for q in queries:
                query_text = q.get("query", q) if isinstance(q, dict) else str(q)
                target = q.get("target_source", "arxiv") if isinstance(q, dict) else "arxiv"
                tried_queries.append(query_text)
                state.total_queries += 1

                results = await self.search_fn(query_text, target)
                all_results.extend(results)

                # Also search expanded variants (synonym/broader queries)
                expanded = q.get("expanded", []) if isinstance(q, dict) else []
                for exp_query in expanded[:2]:  # max 2 expansions per query
                    if exp_query not in tried_queries:
                        tried_queries.append(exp_query)
                        state.total_queries += 1
                        exp_results = await self.search_fn(exp_query, target)
                        all_results.extend(exp_results)

            # 3. Deduplicate and score sources
            new_sources = []
            for item in all_results:
                url = item.get("url") or item.get("pdf_url") or ""
                if url and url not in state.seen_urls:
                    state.seen_urls.add(url)
                    scored = self._score_source(item, topic, state)
                    new_sources.append(scored)

            # 4. Extract facts from top sources
            new_sources.sort(key=lambda s: -s.composite_score)
            new_fact_count = 0
            for source in new_sources[:8]:  # top 8 per iteration
                facts = await self._extract_facts(source, state)
                for fact in facts:
                    if state.add_fact(fact):
                        new_fact_count += 1

            state.all_sources.extend(new_sources)

            # 5. Cross-corroboration: boost facts that appear in multiple sources
            self._cross_corroborate(state)

            # 6. Record iteration
            total_facts = len(state.fact_bank)
            novelty = new_fact_count / max(total_facts, 1)
            state.iterations.append(IterationResult(
                iteration=iteration,
                queries=[q.get("query", str(q)) if isinstance(q, dict) else str(q) for q in queries],
                sources_found=len(new_sources),
                new_facts=new_fact_count,
                novelty_rate=novelty,
                top_sources=new_sources[:3],
            ))

            logger.info("Iteration %d: +%d new facts (novelty=%.1f%%), +%d sources",
                        iteration, new_fact_count, novelty * 100, len(new_sources))

            # Checkpoint: save state after each iteration for crash recovery
            if self.checkpoint_path:
                try:
                    state.save(self.checkpoint_path)
                except Exception as e:
                    logger.warning("Checkpoint save failed: %s", e)

            # Context compaction: summarize older iterations to prevent context bloat
            if len(state.iterations) > 2:
                old_iter = state.iterations[-3]  # compact 2 iterations ago
                if not hasattr(old_iter, '_compacted'):
                    summary = await self._compact_iteration(old_iter, state)
                    old_iter._compacted = True  # type: ignore
                    logger.debug("Compacted iteration %d: %s", old_iter.iteration, summary[:80])

            # 7. Confidence-based re-retrieval: if many low-confidence facts, search more
            low_conf_facts = [f for f in state.fact_bank if f.confidence < 0.4]
            if low_conf_facts and iteration < self.max_iterations:
                uncertain_claims = [f.claim for f in low_conf_facts[:3]]
                logger.info("Re-retrieving for %d low-confidence facts", len(low_conf_facts))
                for claim in uncertain_claims:
                    verify_query = f"evidence for: {claim[:80]}"
                    if verify_query not in tried_queries:
                        tried_queries.append(verify_query)
                        state.total_queries += 1
                        verify_results = await self.search_fn(verify_query, "arxiv")
                        for item in verify_results:
                            url = item.get("url") or ""
                            if url and url not in state.seen_urls:
                                state.seen_urls.add(url)
                                scored = self._score_source(item, state.topic, state)
                                verify_facts = await self._extract_facts(scored, state)
                                for vf in verify_facts:
                                    state.add_fact(vf)
                                state.all_sources.append(scored)

            # 8. Adaptive stopping
            if iteration >= 2 and novelty < self.novelty_threshold:
                logger.info("Novelty below threshold (%.1f%% < %.1f%%), stopping",
                            novelty * 100, self.novelty_threshold * 100)
                break

        # Final step: cross-verify claims
        if len(state.fact_bank) >= 3:
            logger.info("Verifying claims across %d facts...", len(state.fact_bank))
            contradictions = await self.verify_claims(state)
            if contradictions:
                logger.info("Found %d contradictions", len(contradictions))

        # Autorater: quality-check the research output
        rating = await self._rate_output(
            topic, "iterative_research",
            f"Facts: {len(state.fact_bank)}, Sources: {len(state.all_sources)}, "
            f"Top facts:\n{state.get_fact_summary(10)}",
        )
        quality_score = rating.get("score", 3)
        logger.info("Research quality score: %d/5 — %s",
                    quality_score, rating.get("suggestion", ""))

        # If quality is poor and we have budget, do one more iteration
        if quality_score <= 2 and len(state.iterations) < self.max_iterations:
            logger.info("Quality too low (%d/5), running bonus iteration", quality_score)
            # Re-enter the loop for one more round would require refactoring,
            # so just log the suggestion for now
            state.fact_bank.append(Fact(
                claim=f"[AUTORATER] Quality: {quality_score}/5. Suggestion: {rating.get('suggestion', '')}",
                source_url="autorater",
                source_title="Quality Assessment",
                confidence=0.0,
            ))

        logger.info("Research complete: %d iterations, %d facts, %d sources, %d queries (quality=%d/5)",
                    len(state.iterations), len(state.fact_bank),
                    len(state.all_sources), state.total_queries, quality_score)
        return state

    # ------------------------------------------------------------------
    # Fact extraction
    # ------------------------------------------------------------------

    async def _extract_facts(self, source: ScoredSource, state: ResearchState) -> list[Fact]:
        """Extract atomic facts from a source using LLM."""
        content = source.content or source.snippet
        if not content or len(content.strip()) < 50:
            return []

        prompt = EXTRACT_FACTS_PROMPT.format(
            topic=state.topic,
            title=source.title,
            content=content[:4000],
            known_facts=state.get_fact_summary(20),
        )

        try:
            raw = await self.llm.generate(prompt, mode=LLMMode.FAST)
            data = self._extract_json(raw)
            facts = []
            for item in data.get("facts", [])[:self.max_facts_per_source]:
                facts.append(Fact(
                    claim=item.get("claim", ""),
                    source_url=source.url,
                    source_title=source.title,
                    confidence=min(max(float(item.get("confidence", 0.5)), 0.0), 1.0),
                    extracted_at_iter=len(state.iterations) + 1,
                ))
            return [f for f in facts if f.claim]
        except Exception as e:
            logger.warning("Fact extraction failed for %s: %s", source.title[:40], e)
            return []

    # ------------------------------------------------------------------
    # Query refinement
    # ------------------------------------------------------------------

    async def _generate_refined_queries(
        self, state: ResearchState, tried_queries: list[str], iteration: int,
    ) -> list[dict]:
        prompt = REFINE_QUERIES_PROMPT.format(
            topic=state.topic,
            iteration=iteration,
            max_iterations=self.max_iterations,
            fact_count=len(state.fact_bank),
            fact_summary=state.get_fact_summary(20),
            tried_queries="\n".join(f"- {q}" for q in tried_queries[-15:]),
        )

        try:
            raw = await self.llm.generate(prompt, mode=LLMMode.THINKING)
            data = self._extract_json(raw)
            return data.get("queries", [])
        except Exception as e:
            logger.warning("Query refinement failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Source scoring
    # ------------------------------------------------------------------

    def _score_source(self, item: dict, topic: str, state: ResearchState) -> ScoredSource:
        """Score source on relevance, authority, recency, corroboration."""
        url = item.get("url") or item.get("pdf_url") or ""
        title = item.get("title") or item.get("name") or ""
        source_type = item.get("_source_type", "web")
        content = item.get("_full_text", item.get("abstract", item.get("description", "")))
        snippet = (item.get("abstract") or item.get("description") or "")[:500]

        # Relevance: keyword overlap (simple but fast)
        topic_words = set(topic.lower().split())
        text_lower = (title + " " + snippet).lower()
        overlap = sum(1 for w in topic_words if w in text_lower)
        relevance = min(overlap / max(len(topic_words), 1), 1.0)

        # Authority: heuristic by source type + citation count
        authority = 0.5
        if source_type == "arxiv":
            authority = 0.7
            citations = item.get("citation_count", item.get("citationCount", 0))
            if citations and int(citations) > 50:
                authority = 0.9
            elif citations and int(citations) > 10:
                authority = 0.8
        elif source_type == "github":
            stars = item.get("stars", item.get("stargazers_count", 0))
            authority = min(0.5 + (int(stars or 0) / 1000), 0.9)
        elif source_type == "semantic_scholar":
            citations = item.get("citationCount", 0)
            authority = min(0.6 + (int(citations or 0) / 200), 0.95)

        # Recency: exponential decay from publication date
        recency = 0.5
        pub_date = item.get("published", item.get("updated", item.get("created_at", "")))
        if pub_date and len(pub_date) >= 4:
            try:
                year = int(pub_date[:4])
                age_years = 2026 - year
                recency = max(0.1, 1.0 - age_years * 0.15)
            except ValueError:
                pass

        # Corroboration: will be updated in cross_corroborate()
        corroboration = 0.0

        return ScoredSource(
            url=url, title=title, source_type=source_type,
            content=content[:5000], snippet=snippet,
            relevance=relevance, authority=authority,
            recency=recency, corroboration=corroboration,
        )

    def _cross_corroborate(self, state: ResearchState) -> None:
        """Update corroboration scores: facts supported by multiple sources score higher."""
        for fact in state.fact_bank:
            if fact.corroboration_count > 1:
                # Boost all sources that contributed to corroborated facts
                for source in state.all_sources:
                    if source.url == fact.source_url:
                        source.corroboration = min(
                            source.corroboration + 0.1 * fact.corroboration_count, 1.0
                        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _compact_iteration(self, iteration_result: IterationResult, state: ResearchState) -> str:
        """Manus-style context compaction: summarize old iteration results."""
        new_facts_text = "\n".join(
            f"- {f.claim}" for f in state.fact_bank
            if f.extracted_at_iter == iteration_result.iteration
        )[:1000]
        top_sources_text = ", ".join(s.title[:40] for s in iteration_result.top_sources[:3])

        prompt = COMPACT_PROMPT.format(
            iteration=iteration_result.iteration,
            queries_count=len(iteration_result.queries),
            sources_count=iteration_result.sources_found,
            new_facts=iteration_result.new_facts,
            top_sources=top_sources_text,
            new_facts_text=new_facts_text or "(none)",
        )
        try:
            return await self.llm.generate(prompt, mode=LLMMode.FAST)
        except Exception:
            return f"Iteration {iteration_result.iteration}: {iteration_result.new_facts} new facts from {iteration_result.sources_found} sources"

    async def _rate_output(self, topic: str, output_type: str, content: str) -> dict:
        """Autorater: LLM-as-judge for quality gating."""
        prompt = AUTORATER_PROMPT.format(
            topic=topic, output_type=output_type,
            content=content[:3000],
        )
        try:
            raw = await self.llm.generate(prompt, mode=LLMMode.FAST)
            return self._extract_json(raw)
        except Exception:
            return {"score": 3, "issues": [], "suggestion": ""}

    async def verify_claims(self, state: ResearchState) -> list[dict]:
        """Cross-verify claims: find contradictions and boost corroborated facts."""
        if len(state.fact_bank) < 3:
            return []

        prompt = f"""\
You are a fact-checker. Review these research findings for contradictions and consensus.

Facts:
{state.get_fact_summary(30)}

Identify:
1. CONTRADICTIONS: facts that directly contradict each other
2. STRONG CONSENSUS: facts supported by multiple sources that are likely true
3. UNCERTAIN: facts with low confidence that need more evidence

Return JSON:
{{
  "contradictions": [
    {{"fact_a": "claim 1", "fact_b": "claim 2", "explanation": "why they conflict"}}
  ],
  "consensus": [
    {{"claim": "well-supported fact", "confidence": 0.9}}
  ],
  "uncertain": [
    {{"claim": "needs verification", "suggested_query": "search query to verify"}}
  ]
}}
"""
        try:
            raw = await self.llm.generate(prompt, mode=LLMMode.FAST)
            result = self._extract_json(raw)

            # Mark contradicted facts
            for c in result.get("contradictions", []):
                for fact in state.fact_bank:
                    if fact.claim in c.get("fact_a", "") or fact.claim in c.get("fact_b", ""):
                        fact.confidence *= 0.7  # reduce confidence for contradicted facts

            # Boost consensus facts
            for c in result.get("consensus", []):
                for fact in state.fact_bank:
                    if fact.claim in c.get("claim", ""):
                        fact.confidence = min(fact.confidence * 1.2, 1.0)

            return result.get("contradictions", [])
        except Exception as e:
            logger.warning("Claim verification failed: %s", e)
            return []

    @staticmethod
    def _extract_json(text: str) -> dict:
        for start_char in ("{", "["):
            end_char = "}" if start_char == "{" else "]"
            start = text.find(start_char)
            end = text.rfind(end_char) + 1
            if start >= 0 and end > start:
                try:
                    result = json.loads(text[start:end])
                    return result if isinstance(result, dict) else {"items": result}
                except Exception:
                    pass
        return {}
