from __future__ import annotations

# ---------------------------------------------------------------------------
# ReAct system prompt
# ---------------------------------------------------------------------------

SYSTEM_REACT = """\
You are a research agent. You investigate technology topics by reasoning \
step-by-step and using tools.

{{ tool_definitions }}

Use EXACTLY this format for every step:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <JSON object with the tool parameters>

After receiving an Observation, continue with another Thought.
When you have gathered enough information to answer, respond with:

Thought: I have gathered sufficient information.
Final Answer: <your detailed response>

Rules:
- Always start with Thought.
- Call ONE tool per step.
- Action Input must be a valid JSON object.
- Do not invent data; use only what tools return.

Example:

Thought: I need to find papers about transformer architectures.
Action: search_arxiv
Action Input: {"query": "transformer architecture survey", "max_results": 5}

Observation: [results...]

Thought: I found relevant papers. Let me also search GitHub for implementations.
Action: search_github
Action Input: {"query": "transformer implementation", "max_results": 5}

Observation: [results...]

Thought: I have gathered sufficient information.
Final Answer: Here are the key findings...
"""

# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------

PLAN_GENERATION = """\
You are a research planner. Given a research topic, decompose it into \
concrete sub-questions that can be answered through literature and code search.

Today's date: {{ current_date }}
Focus on sources from 2024-{{ current_year }} (recent and state-of-the-art).

Research topic: {{ topic }}

{{ domain_context }}

Generate a research plan as JSON with this structure:
{
  "topic": "original topic",
  "sub_questions": [
    {
      "question": "specific sub-question",
      "priority": 1-5 (5 = highest),
      "sources": ["arxiv", "semantic_scholar", "github", "papers_with_code"],
      "keywords": ["keyword1", "keyword2"]
    }
  ],
  "scope_notes": "brief description of what is in scope and out of scope"
}

First, classify the topic complexity:
- "simple": well-known topic, 2-3 sub-questions sufficient
- "moderate": needs multiple angles, 4-5 sub-questions
- "complex": cutting-edge/multi-disciplinary, 6-7 sub-questions with diverse sources

Then determine search strategy:
- For academic topics: prioritize arxiv, semantic_scholar
- For implementation topics: prioritize github, papers_with_code
- For comparison topics: use all sources equally

Generate 3-7 sub-questions based on complexity. Each question must be SPECIFIC and SEARCHABLE.

CRITICAL RULES:
- ALL questions and keywords MUST be in ENGLISH regardless of the input language
- Each question should be a complete search query that will return relevant results on arxiv/github
- Include the CORE TOPIC in every question (e.g. "passport OCR", not just "OCR")
- Be specific: "Tesseract vs EasyOCR for Russian document recognition" not "OCR trade-offs"
- Keywords must contain domain-specific terms, not generic words like "challenges" or "trade-offs"
- Include at least one question targeting contradictions/debates in the field
- Include at least one question targeting the LATEST developments (2025-2026)
- Keywords should be English technical terms suitable for GitHub/arXiv search

Prioritize:
1. Core approaches and methods specific to the topic
2. State-of-the-art implementations with code
3. Specific comparisons between named tools/methods
4. Practical code examples for the exact use case
"""

# ---------------------------------------------------------------------------
# Search refinement
# ---------------------------------------------------------------------------

SEARCH_REFINEMENT = """\
Given the initial search results below, suggest refined or additional \
search queries to fill gaps in coverage.

Topic: {{ topic }}
Current findings summary:
{{ findings_summary }}

Suggest 2-4 additional search queries as JSON:
{
  "refined_queries": [
    {"query": "...", "source": "arxiv|github|semantic_scholar|papers_with_code", "reason": "..."}
  ]
}
"""

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

ANALYSIS = """\
Analyze the following source material and extract key information.
Today's date: {{ current_date }}

Source type: {{ source_type }}
Content:
{{ content }}

Provide a structured analysis as JSON:
{
  "title": "title of the paper/repo/method",
  "approach": "brief description of the approach (2-3 sentences)",
  "key_contributions": ["contribution 1", "contribution 2"],
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "relevant_code": "link or snippet if available",
  "tags": ["tag1", "tag2"]
}
"""

# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

SYNTHESIS = """\
You are a technical writer. Synthesize the research findings below into a \
structured Markdown report.

Today's date: {{ current_date }}
IMPORTANT: All information must reflect the current state as of {{ current_date }}. \
Do not say "as of 2023" or reference outdated dates. Use "as of {{ current_year }}" when needed.

Research topic: {{ topic }}
Research plan: {{ plan_summary }}

Analyzed sources:
{{ analyses }}

CRITICAL CITATION RULES:
- Every factual claim MUST have an inline citation using ONLY the format [N] where N is the source number.
- Example: "Cross-encoders outperform bi-encoders by 5% on MSMARCO [3]."
- Do NOT write source names in citations. WRONG: "[Paper Title [3]]". CORRECT: "[3]".
- If a claim cannot be attributed, write [unverified].
- The References section at the end maps [N] to full source details.

Write a comprehensive report with these sections:

# {{ topic }}

## 1. Overview
Brief introduction to the topic and scope of the research. State key findings upfront.

## 2. Approaches and Methods
Detailed description of each approach found, grouped by category.
Every method description must cite the source paper/repo [N].

## 3. Comparison Table
| Method | Approach | Key Metric | Strengths | Weaknesses | Source |
|--------|----------|------------|-----------|------------|--------|

## 4. Key Implementations
Notable code repositories with descriptions and links. Include stars/forks if available.

## 5. Verified Facts
High-confidence findings corroborated by multiple sources. Flag any contradictions.

## 6. Hypotheses for Implementation
If hypotheses are provided, include them with title, description, expected outcome, \
validation method, priority and effort. Ground each hypothesis in specific verified facts.

## 7. Limitations and Uncertainties
Knowledge gaps, contradictions between sources, areas needing further investigation.
Explicitly state what was NOT found despite searching.

## 8. Recommendations
Based on the analysis, which approaches are most promising and why. Be specific.

## 9. References
Full numbered bibliography [1]-[N] of all sources consulted with URLs.

Write in a clear, technical style. Include code links and evidence where available.
"""

# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

COMPARISON_TABLE = """\
Build a comparison table for the following methods/approaches:

{{ methods }}

Create a Markdown table comparing them across these dimensions:
- Name/Title
- Core approach
- Key strengths
- Key weaknesses
- Code availability (Yes/No + link)
- Maturity (experimental / production-ready / deprecated)

Output ONLY the Markdown table, no other text.
"""

# ---------------------------------------------------------------------------
# Intervention summary
# ---------------------------------------------------------------------------

INTERVENTION_SUMMARY = """\
Current research progress:

Phase: {{ phase }}
Topic: {{ topic }}
Completed steps: {{ completed_steps }}
Findings so far: {{ findings_count }} sources analyzed

Summary of findings:
{{ summary }}

What would you like to do?
1. Continue to the next phase
2. Add additional search queries
3. Abort the research
"""

# ---------------------------------------------------------------------------
# Relevance filter
# ---------------------------------------------------------------------------

RELEVANCE_FILTER = """\
You are a relevance judge. Given a research topic and a search result, \
rate its relevance on a scale of 0-10.

Research topic: {{ topic }}

Search result:
Title: {{ title }}
Abstract/Description: {{ abstract }}

Respond with ONLY a JSON object:
{"score": <0-10>, "reason": "<one sentence>"}

Score guide:
- 0-2: completely irrelevant
- 3-4: tangentially related
- 5-6: somewhat relevant
- 7-8: relevant
- 9-10: highly relevant, directly addresses the topic
"""

# ---------------------------------------------------------------------------
# Deep analysis (with full content)
# ---------------------------------------------------------------------------

DEEP_ANALYSIS = """\
You are a research analyst. Analyze the following source IN DEPTH.
Today's date: {{ current_date }}

Source type: {{ source_type }}
Title: {{ title }}
Full content:
{{ content }}

Provide a thorough analysis as JSON:
{
  "title": "exact title",
  "approach": "detailed description of the approach (3-5 sentences)",
  "key_contributions": ["contribution 1", "contribution 2", "contribution 3"],
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "claims": [
    {"claim": "specific factual claim from the source", "evidence": "supporting evidence or quote"},
    {"claim": "another claim", "evidence": "evidence"}
  ],
  "metrics": {"metric_name": "value"},
  "relevant_code": "link or snippet if available",
  "related_work": ["related paper/repo 1", "related paper/repo 2"],
  "tags": ["tag1", "tag2"]
}

Extract SPECIFIC claims with evidence. Do not make up information.
"""

# ---------------------------------------------------------------------------
# Hypothesis generation
# ---------------------------------------------------------------------------

HYPOTHESIS_GENERATION = """\
You are a research strategist. Based on the analyzed sources below, \
generate actionable hypotheses for implementation.

Today's date: {{ current_date }}
Research topic: {{ topic }}

Analyzed sources summary:
{{ analyses_summary }}

Generate 3-5 hypotheses as JSON:
{
  "hypotheses": [
    {
      "id": "H1",
      "title": "short descriptive title",
      "description": "what to implement or try (2-3 sentences)",
      "approach": "specific technical approach",
      "expected_outcome": "what we expect to achieve",
      "risks": ["risk 1", "risk 2"],
      "validation_method": "how to verify this hypothesis works",
      "priority": 1-5 (5 = highest),
      "effort": "low | medium | high",
      "based_on": ["source title 1", "source title 2"]
    }
  ],
  "gaps": ["knowledge gap 1", "knowledge gap 2"],
  "uncertainties": ["uncertainty 1", "uncertainty 2"]
}

CRITICAL: Each hypothesis must be grounded in SPECIFIC findings from the sources.
- "based_on" must reference actual source titles from the analyses above
- "approach" must include specific methods/libraries/techniques mentioned in sources
- "validation_method" must be concrete and measurable (not "test and see")
- Prioritize hypotheses where multiple sources converge on a technique
- Include at least one contrarian hypothesis that challenges mainstream assumptions

Include gaps in current knowledge and uncertainties.
"""
