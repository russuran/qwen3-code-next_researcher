"""AIDE-style solution tree for tracking experiment branches.

Each hypothesis experiment is a node. Benchmark results + structured
reflections drive the tree search: refine winners, debug failures,
or draft fresh ideas informed by all prior results.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class BranchType(str, Enum):
    SEED = "seed"       # initial hypothesis from research
    REFINE = "refine"   # improve best-performing node
    DEBUG = "debug"     # fix broken/failing node
    DRAFT = "draft"     # fresh idea informed by all prior results
    MERGE = "merge"     # combine best aspects of two+ nodes


class Reflection(BaseModel):
    loss_diagnosis: str = ""
    what_worked: list[str] = Field(default_factory=list)
    what_failed: list[str] = Field(default_factory=list)
    suggested_next: BranchType = BranchType.REFINE
    confidence: float = 0.5


class ExperimentNode(BaseModel):
    id: str
    parent_id: str | None = None
    hypothesis: dict = Field(default_factory=dict)
    branch_type: BranchType = BranchType.SEED
    depth: int = 0
    metrics: dict = Field(default_factory=dict)
    reflection: Reflection | None = None
    hyp_params: dict = Field(default_factory=dict)
    code_path: str = ""
    repo_branch: str = ""
    status: str = "pending"  # pending / running / completed / failed / killed
    error: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def val_loss(self) -> float | None:
        return self.metrics.get("val_loss")

    @property
    def title(self) -> str:
        return self.hypothesis.get("title", self.id)


class SolutionTree(BaseModel):
    nodes: dict[str, ExperimentNode] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        hypothesis: dict,
        branch_type: BranchType = BranchType.SEED,
        parent_id: str | None = None,
        hyp_params: dict | None = None,
    ) -> ExperimentNode:
        depth = 0
        if parent_id and parent_id in self.nodes:
            depth = self.nodes[parent_id].depth + 1
        node = ExperimentNode(
            id=node_id,
            parent_id=parent_id,
            hypothesis=hypothesis,
            branch_type=branch_type,
            depth=depth,
            hyp_params=hyp_params or {},
        )
        self.nodes[node_id] = node
        return node

    def update_metrics(self, node_id: str, metrics: dict) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].metrics = metrics
            self.nodes[node_id].status = "completed"

    def mark_failed(self, node_id: str, error: str = "") -> None:
        if node_id in self.nodes:
            self.nodes[node_id].status = "failed"
            self.nodes[node_id].error = error

    def set_reflection(self, node_id: str, reflection: Reflection) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].reflection = reflection

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_best(self, n: int = 3) -> list[ExperimentNode]:
        """Top-n nodes by lowest val_loss (completed only)."""
        completed = [
            nd for nd in self.nodes.values()
            if nd.status == "completed" and nd.val_loss is not None
        ]
        return sorted(completed, key=lambda nd: nd.val_loss)[:n]

    def get_failed(self) -> list[ExperimentNode]:
        return [nd for nd in self.nodes.values() if nd.status == "failed"]

    def get_lineage(self, node_id: str) -> list[ExperimentNode]:
        """Walk from root to *node_id*."""
        chain: list[ExperimentNode] = []
        cur = self.nodes.get(node_id)
        while cur:
            chain.append(cur)
            cur = self.nodes.get(cur.parent_id) if cur.parent_id else None
        return list(reversed(chain))

    def completed_count(self) -> int:
        return sum(1 for nd in self.nodes.values() if nd.status == "completed")

    # ------------------------------------------------------------------
    # Context for LLM
    # ------------------------------------------------------------------

    def build_context_prompt(self, max_nodes: int = 15) -> str:
        """Serialize experiment history for the LLM (≤300 tok per node).

        For large trees (>max_nodes): show top-5 best + bottom-3 worst +
        last 3 recent + all failed. This keeps context bounded while
        preserving the most informative experiments.
        """
        lines: list[str] = []
        completed = sorted(
            [nd for nd in self.nodes.values() if nd.status == "completed" and nd.val_loss is not None],
            key=lambda nd: nd.val_loss,
        )
        failed = [nd for nd in self.nodes.values() if nd.status == "failed"]

        if len(completed) + len(failed) > max_nodes:
            # Large tree: smart selection
            top5 = completed[:5]
            bottom3 = completed[-3:] if len(completed) > 5 else []
            recent3 = sorted(completed, key=lambda nd: nd.created_at)[-3:]
            # Deduplicate while preserving order
            seen = set()
            selected: list[ExperimentNode] = []
            for nd in top5 + recent3 + bottom3 + failed[:3]:
                if nd.id not in seen:
                    seen.add(nd.id)
                    selected.append(nd)
        else:
            selected = completed + failed

        for i, nd in enumerate(selected[:max_nodes]):
            tag = f"[{nd.branch_type.value}]"
            loss_str = f"val_loss={nd.val_loss:.4f}" if nd.val_loss is not None else "FAILED"
            ref = ""
            if nd.reflection:
                ref = f"\n    Reflection: {nd.reflection.loss_diagnosis[:150]}"
                if nd.reflection.what_worked:
                    ref += f"\n    Worked: {', '.join(nd.reflection.what_worked[:3])}"
                if nd.reflection.what_failed:
                    ref += f"\n    Failed: {', '.join(nd.reflection.what_failed[:3])}"
            parent = f" (parent: {nd.parent_id})" if nd.parent_id else ""
            params = ""
            if nd.hyp_params:
                params = f"\n    Params: {json.dumps({k: v for k, v in nd.hyp_params.items()}, default=str)}"
            lines.append(
                f"  {i+1}. {tag} {nd.title[:60]} — {loss_str}{parent}"
                f"{params}{ref}"
            )

        if not lines:
            return "(no experiments yet)"
        return "Experiment history (sorted by val_loss, best first):\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            self.model_dump_json(indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> SolutionTree:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
