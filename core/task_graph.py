"""DAG-based task graph for execution planning."""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    CLARIFY = "clarify"
    SEARCH = "search"
    FETCH = "fetch"
    PARSE = "parse"
    EXTRACT = "extract"
    ANALYZE = "analyze"
    COMPARE = "compare"
    SANDBOX_RUN = "sandbox_run"
    SYNTHESIZE = "synthesize"
    EVALUATE = "evaluate"
    PUBLISH = "publish"


class NodeStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    node_type: NodeType
    status: NodeStatus = NodeStatus.PENDING
    params: dict[str, Any] = {}
    result: Any = None
    error: str | None = None
    depends_on: list[str] = []
    priority: int = 0


class TaskGraph(BaseModel):
    """Directed Acyclic Graph of task nodes."""
    run_id: str = ""
    nodes: dict[str, TaskNode] = {}

    def add_node(self, node: TaskNode) -> TaskNode:
        self.nodes[node.id] = node
        return node

    def add_edge(self, from_id: str, to_id: str) -> None:
        if to_id in self.nodes:
            if from_id not in self.nodes[to_id].depends_on:
                self.nodes[to_id].depends_on.append(from_id)

    def get_ready_nodes(self) -> list[TaskNode]:
        """Return nodes whose dependencies are all completed."""
        ready = []
        for node in self.nodes.values():
            if node.status != NodeStatus.PENDING:
                continue
            deps_met = all(
                self.nodes[dep].status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED)
                for dep in node.depends_on
                if dep in self.nodes
            )
            if deps_met:
                ready.append(node)
        return sorted(ready, key=lambda n: -n.priority)

    def mark_running(self, node_id: str) -> None:
        self.nodes[node_id].status = NodeStatus.RUNNING

    def mark_completed(self, node_id: str, result: Any = None) -> None:
        self.nodes[node_id].status = NodeStatus.COMPLETED
        self.nodes[node_id].result = result

    def mark_failed(self, node_id: str, error: str) -> None:
        self.nodes[node_id].status = NodeStatus.FAILED
        self.nodes[node_id].error = error

    def is_complete(self) -> bool:
        return all(
            n.status in (NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.SKIPPED)
            for n in self.nodes.values()
        )

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for n in self.nodes.values():
            counts[n.status.value] = counts.get(n.status.value, 0) + 1
        return counts
