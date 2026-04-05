from __future__ import annotations

import pytest

from core.task_graph import NodeType, NodeStatus, TaskGraph, TaskNode
from core.scheduler import Scheduler


def test_add_node_and_edge():
    g = TaskGraph()
    n1 = g.add_node(TaskNode(id="a", name="search", node_type=NodeType.SEARCH))
    n2 = g.add_node(TaskNode(id="b", name="analyze", node_type=NodeType.ANALYZE))
    g.add_edge("a", "b")
    assert "a" in g.nodes["b"].depends_on


def test_ready_nodes():
    g = TaskGraph()
    g.add_node(TaskNode(id="a", name="search", node_type=NodeType.SEARCH))
    g.add_node(TaskNode(id="b", name="analyze", node_type=NodeType.ANALYZE, depends_on=["a"]))

    ready = g.get_ready_nodes()
    assert len(ready) == 1
    assert ready[0].id == "a"


def test_ready_after_completion():
    g = TaskGraph()
    g.add_node(TaskNode(id="a", name="search", node_type=NodeType.SEARCH))
    g.add_node(TaskNode(id="b", name="analyze", node_type=NodeType.ANALYZE, depends_on=["a"]))

    g.mark_running("a")
    g.mark_completed("a", result="done")

    ready = g.get_ready_nodes()
    assert len(ready) == 1
    assert ready[0].id == "b"


def test_is_complete():
    g = TaskGraph()
    g.add_node(TaskNode(id="a", name="t1", node_type=NodeType.SEARCH))
    assert not g.is_complete()

    g.mark_running("a")
    g.mark_completed("a")
    assert g.is_complete()


def test_summary():
    g = TaskGraph()
    g.add_node(TaskNode(id="a", name="t1", node_type=NodeType.SEARCH, status=NodeStatus.COMPLETED))
    g.add_node(TaskNode(id="b", name="t2", node_type=NodeType.ANALYZE, status=NodeStatus.FAILED))
    g.add_node(TaskNode(id="c", name="t3", node_type=NodeType.SYNTHESIZE, status=NodeStatus.PENDING))

    s = g.summary()
    assert s["completed"] == 1
    assert s["failed"] == 1
    assert s["pending"] == 1


@pytest.mark.asyncio
async def test_scheduler_runs_graph():
    g = TaskGraph()
    g.add_node(TaskNode(id="a", name="t1", node_type=NodeType.SEARCH))
    g.add_node(TaskNode(id="b", name="t2", node_type=NodeType.ANALYZE, depends_on=["a"]))

    results = []

    async def executor(node: TaskNode):
        results.append(node.id)
        return f"result_{node.id}"

    scheduler = Scheduler(g, executor, max_concurrency=2)
    await scheduler.run()

    assert results == ["a", "b"]
    assert g.nodes["a"].status == NodeStatus.COMPLETED
    assert g.nodes["b"].status == NodeStatus.COMPLETED
