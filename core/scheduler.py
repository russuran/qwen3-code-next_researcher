"""DAG scheduler: picks ready nodes, dispatches to worker pool."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Awaitable

from core.task_graph import NodeStatus, TaskGraph, TaskNode

logger = logging.getLogger(__name__)

NodeExecutor = Callable[[TaskNode], Awaitable[Any]]


class Scheduler:
    def __init__(
        self,
        graph: TaskGraph,
        executor: NodeExecutor,
        max_concurrency: int = 4,
        on_node_start: Callable[[TaskNode], Awaitable[None]] | None = None,
        on_node_complete: Callable[[TaskNode], Awaitable[None]] | None = None,
    ) -> None:
        self.graph = graph
        self.executor = executor
        self.max_concurrency = max_concurrency
        self.on_node_start = on_node_start
        self.on_node_complete = on_node_complete
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run(self) -> TaskGraph:
        """Execute the graph until all nodes are complete."""
        while not self.graph.is_complete():
            ready = self.graph.get_ready_nodes()
            if not ready:
                # All remaining nodes have unmet deps or are running
                running = [n for n in self.graph.nodes.values() if n.status == NodeStatus.RUNNING]
                if not running:
                    logger.warning("Deadlock: no ready or running nodes")
                    break
                await asyncio.sleep(0.1)
                continue

            tasks = [self._execute_node(node) for node in ready]
            await asyncio.gather(*tasks, return_exceptions=True)

        return self.graph

    async def _execute_node(self, node: TaskNode) -> None:
        async with self._semaphore:
            self.graph.mark_running(node.id)
            logger.info("Running node: %s (%s)", node.name, node.node_type.value)

            if self.on_node_start:
                await self.on_node_start(node)

            try:
                result = await self.executor(node)
                self.graph.mark_completed(node.id, result)
                logger.info("Completed node: %s", node.name)
            except Exception as e:
                self.graph.mark_failed(node.id, str(e))
                logger.error("Failed node %s: %s", node.name, e)

            if self.on_node_complete:
                await self.on_node_complete(node)
