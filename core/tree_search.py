"""AIDE-style tree search loop with structured reflections.

After each benchmark, generates a reflection explaining why the result
was good/bad. Prior (hypothesis, metric, reflection) triples are fed
to the LLM when generating the next hypothesis. Three branch types:
refine (improve best), debug (fix broken), draft (fresh idea).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Awaitable

from core.llm import LLM, LLMMode
from core.solution_tree import (
    BranchType, ExperimentNode, Reflection, SolutionTree,
)

logger = logging.getLogger(__name__)

REFLECTION_PROMPT = """\
You just ran an ML experiment. Analyze the results concisely.

Hypothesis: {title}
Approach: {approach}
Branch type: {branch_type}
Metrics: {metrics}
Hyperparameters: {hyp_params}
Error (if any): {error}

All prior experiments:
{tree_context}

Respond with ONLY valid JSON:
{{
  "loss_diagnosis": "One sentence: why was val_loss this value?",
  "what_worked": ["up to 3 things to keep doing"],
  "what_failed": ["up to 3 things to avoid"],
  "suggested_next": "refine|debug|draft",
  "confidence": 0.0
}}
"""

NEXT_HYPOTHESIS_PROMPT = """\
You are an ML researcher deciding what experiment to run next.

Task: {topic}
Repository: {repo_name}

Full experiment history:
{tree_context}

Best result so far: {best_summary}

Branch type for this iteration: {branch_type}
{parent_context}

Rules:
- "refine": take the best node's approach and make a SPECIFIC improvement
- "debug": fix the specific error in the failing node
- "draft": try a completely DIFFERENT approach, informed by what failed before
- "merge": combine the best aspects of the top 2-3 nodes into one approach

Return ONLY valid JSON:
{{
  "title": "Short title (5-8 words)",
  "description": "What to change and why",
  "approach": "Technical approach",
  "target_file": "relative/path/to/file.py (or empty for standalone)",
  "change_description": "Specific code change to make",
  "parent_id": "node_id of parent (null for draft)",
  "expected_outcome": "What metric improvement to expect"
}}
"""


class TreeSearchLoop:
    """Run AIDE-style tree search over ML experiments."""

    def __init__(
        self,
        llm: LLM,
        tree: SolutionTree,
        topic: str,
        repo_context: dict | None = None,
        workspace: str = "",
        max_iterations: int = 5,
        implement_fn: Callable[..., Awaitable[dict]] | None = None,
        benchmark_fn: Callable[..., Awaitable[dict]] | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        self.llm = llm
        self.tree = tree
        self.topic = topic
        self.repo_context = repo_context
        self.workspace = workspace
        self.max_iterations = max_iterations
        self._implement = implement_fn
        self._benchmark = benchmark_fn
        self._progress = on_progress

    async def seed(self, implementations: list[dict]) -> None:
        """Add initial Phase 3 results as SEED nodes."""
        for impl in implementations:
            hyp = impl.get("hypothesis", {})
            node_id = hyp.get("id", hyp.get("title", "")[:20]).replace(" ", "_")
            node = self.tree.add_node(
                node_id=node_id,
                hypothesis=hyp,
                branch_type=BranchType.SEED,
                hyp_params=impl.get("hyp_params", {}),
            )
            node.code_path = impl.get("code_path", "")
            node.repo_branch = impl.get("repo_branch", "")

            metrics = impl.get("benchmark_metrics", {})
            if metrics:
                self.tree.update_metrics(node_id, metrics)
            elif not impl.get("smoke_test_passed"):
                self.tree.mark_failed(node_id, impl.get("error", "smoke test failed"))

    async def run(self) -> SolutionTree:
        """Execute tree search iterations."""
        for iteration in range(1, self.max_iterations + 1):
            if self._progress:
                await self._progress(
                    "tree_search",
                    f"Iteration {iteration}/{self.max_iterations}"
                )

            # 1. Generate reflection for the latest completed node
            latest = self._latest_node()
            if latest and latest.status == "completed" and not latest.reflection:
                reflection = await self._generate_reflection(latest)
                self.tree.set_reflection(latest.id, reflection)

            # 2. Choose branch type
            branch_type, parent_node = self._choose_branch()
            logger.info(
                "Tree search iter %d: branch=%s parent=%s",
                iteration, branch_type.value,
                parent_node.id if parent_node else "none",
            )

            # 3. Generate next hypothesis
            hyp = await self._generate_next_hypothesis(branch_type, parent_node)
            if not hyp:
                logger.warning("LLM returned no hypothesis, stopping tree search")
                break

            node_id = f"iter{iteration}_{branch_type.value}"
            node = self.tree.add_node(
                node_id=node_id,
                hypothesis=hyp,
                branch_type=branch_type,
                parent_id=parent_node.id if parent_node else None,
            )

            # 4. Implement
            if self._implement:
                impl = await self._implement(
                    hyp,
                    "transformers,peft,datasets,torch,mlx-lm",
                    repo_context=self.repo_context,
                )
                node.code_path = impl.get("code_path", "")
                node.repo_branch = impl.get("repo_branch", "")

                if not impl.get("smoke_test_passed"):
                    self.tree.mark_failed(node_id, "smoke test failed")
                    continue

            # 5. Benchmark
            if self._benchmark and node.code_path:
                from pathlib import Path
                impl_dir = Path(node.code_path).parent
                metrics = await self._benchmark(impl_dir)
                self.tree.update_metrics(node_id, metrics)

                val_loss = metrics.get("val_loss")
                logger.info(
                    "Tree search iter %d: %s → val_loss=%s",
                    iteration, hyp.get("title", ""), val_loss,
                )

            # 6. Check early termination
            if self._should_stop():
                logger.info("Tree search: early stop (plateau detected)")
                break

        # Final reflections for any un-reflected completed nodes
        for node in self.tree.nodes.values():
            if node.status == "completed" and not node.reflection:
                ref = await self._generate_reflection(node)
                self.tree.set_reflection(node.id, ref)

        return self.tree

    # ------------------------------------------------------------------
    # Reflection
    # ------------------------------------------------------------------

    async def _generate_reflection(self, node: ExperimentNode) -> Reflection:
        metrics_str = json.dumps(node.metrics, default=str)[:300]
        params_str = json.dumps(node.hyp_params, default=str)[:300]

        prompt = REFLECTION_PROMPT.format(
            title=node.title,
            approach=node.hypothesis.get("approach", ""),
            branch_type=node.branch_type.value,
            metrics=metrics_str,
            hyp_params=params_str,
            error=node.error[:200],
            tree_context=self.tree.build_context_prompt(),
        )
        raw = await self.llm.generate(prompt, mode=LLMMode.FAST)

        try:
            data = self._extract_json(raw)
            return Reflection(
                loss_diagnosis=data.get("loss_diagnosis", ""),
                what_worked=data.get("what_worked", []),
                what_failed=data.get("what_failed", []),
                suggested_next=BranchType(data.get("suggested_next", "refine")),
                confidence=float(data.get("confidence", 0.5)),
            )
        except Exception:
            return Reflection(loss_diagnosis="(reflection parse failed)")

    # ------------------------------------------------------------------
    # Branch selection heuristic
    # ------------------------------------------------------------------

    def _choose_branch(self) -> tuple[BranchType, ExperimentNode | None]:
        """Heuristic: refine if improving, debug if broken, draft if stuck."""
        best = self.tree.get_best(1)
        failed = self.tree.get_failed()

        # If there's a recent failure with a fixable error → debug
        if failed:
            last_fail = failed[-1]
            if last_fail.error and "timeout" not in last_fail.error.lower():
                return BranchType.DEBUG, last_fail

        if not best:
            return BranchType.DRAFT, None

        best_node = best[0]

        # Check if we've been plateauing
        completed = [
            nd for nd in self.tree.nodes.values()
            if nd.status == "completed" and nd.val_loss is not None
        ]
        if len(completed) >= 3:
            recent_3 = sorted(completed, key=lambda nd: nd.created_at)[-3:]
            losses = [nd.val_loss for nd in recent_3]
            # If last 3 losses are within 5% of each other → stuck
            if losses and max(losses) - min(losses) < 0.05 * min(losses):
                # Try MERGE first (combine top-2), then DRAFT if already merged
                recent_types = [nd.branch_type for nd in recent_3]
                if BranchType.MERGE not in recent_types and len(self.tree.get_best(2)) >= 2:
                    return BranchType.MERGE, best_node
                return BranchType.DRAFT, None

        # Check if last reflection suggested something specific
        if best_node.reflection and best_node.reflection.suggested_next:
            return best_node.reflection.suggested_next, best_node

        # Default: refine the best
        return BranchType.REFINE, best_node

    # ------------------------------------------------------------------
    # Next hypothesis generation
    # ------------------------------------------------------------------

    async def _generate_next_hypothesis(
        self, branch_type: BranchType, parent: ExperimentNode | None,
    ) -> dict | None:
        parent_context = ""
        if branch_type == BranchType.MERGE:
            # For merge: show top-2 nodes to combine
            top2 = self.tree.get_best(2)
            parent_context = "Nodes to MERGE (combine best aspects of both):\n"
            for nd in top2:
                parent_context += (
                    f"  - {nd.title}: val_loss={nd.val_loss}\n"
                    f"    Params: {json.dumps(nd.hyp_params, default=str)[:200]}\n"
                )
                if nd.reflection:
                    parent_context += f"    Worked: {nd.reflection.what_worked}\n"
        elif parent:
            parent_context = (
                f"Parent node to build on:\n"
                f"  Title: {parent.title}\n"
                f"  Metrics: {json.dumps(parent.metrics, default=str)[:200]}\n"
                f"  Params: {json.dumps(parent.hyp_params, default=str)[:200]}\n"
            )
            if parent.reflection:
                parent_context += (
                    f"  Reflection: {parent.reflection.loss_diagnosis}\n"
                    f"  Worked: {parent.reflection.what_worked}\n"
                    f"  Failed: {parent.reflection.what_failed}\n"
                )
            if parent.error:
                parent_context += f"  Error: {parent.error[:200]}\n"

        best = self.tree.get_best(1)
        best_summary = "none yet"
        if best:
            b = best[0]
            best_summary = f"{b.title} — val_loss={b.val_loss}, params={json.dumps(b.hyp_params, default=str)[:150]}"

        prompt = NEXT_HYPOTHESIS_PROMPT.format(
            topic=self.topic,
            repo_name=self.repo_context.get("repo_path", "").split("/")[-1] if self.repo_context else "",
            tree_context=self.tree.build_context_prompt(),
            best_summary=best_summary,
            branch_type=branch_type.value,
            parent_context=parent_context or "(no parent — this is a fresh draft)",
        )

        raw = await self.llm.generate(prompt, mode=LLMMode.THINKING)
        hyp = self._extract_json(raw)
        if not hyp or "title" not in hyp:
            return None

        hyp["source"] = f"tree_search_{branch_type.value}"
        return hyp

    # ------------------------------------------------------------------
    # Early termination
    # ------------------------------------------------------------------

    def _should_stop(self) -> bool:
        """Stop if last N completed nodes show no improvement.

        For long runs (50+ iterations) we allow more plateau tolerance:
        must see 5 consecutive non-improving nodes, and must have tried
        both MERGE and DRAFT before giving up.
        """
        completed = sorted(
            [nd for nd in self.tree.nodes.values()
             if nd.status == "completed" and nd.val_loss is not None],
            key=lambda nd: nd.created_at,
        )
        # Need seeds + at least a few iterations
        plateau_window = max(5, min(len(completed) // 3, 10))
        if len(completed) < plateau_window + 1:
            return False

        recent = completed[-plateau_window:]
        best_ever = min(nd.val_loss for nd in completed)
        recent_best = min(nd.val_loss for nd in recent)

        if recent_best < best_ever * 1.001:
            return False  # still improving

        # Before stopping, check we've tried MERGE and DRAFT
        recent_types = {nd.branch_type for nd in recent}
        if BranchType.MERGE not in recent_types or BranchType.DRAFT not in recent_types:
            return False  # haven't exhausted strategies yet

        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _latest_node(self) -> ExperimentNode | None:
        nodes = sorted(self.tree.nodes.values(), key=lambda nd: nd.created_at)
        return nodes[-1] if nodes else None

    @staticmethod
    def _extract_json(text: str) -> dict:
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char) + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except Exception:
                    pass
        return {}
