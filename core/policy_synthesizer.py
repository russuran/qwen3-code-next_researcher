"""Policy synthesizer: builds execution policy from task model and config."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from core.task_model_builder import TaskModel

logger = logging.getLogger(__name__)


class RetryPolicy(BaseModel):
    max_retries: int = 2
    backoff_factor: float = 1.5


class ExecutionPolicy(BaseModel):
    parallelism: int = 4
    timeout_sec: int = 600
    retry: RetryPolicy = RetryPolicy()
    allow_network: bool = True
    sandbox_enabled: bool = False
    max_sources: int = 20


_RISK_ADJUSTMENTS: dict[str, dict[str, Any]] = {
    "high_dependency_count": {"parallelism": 2, "timeout_sec": 900},
    "sandbox": {"sandbox_enabled": True, "timeout_sec": 1200},
    "code_changes": {"sandbox_enabled": True, "parallelism": 2},
}


def synthesize_policy(
    task_model: TaskModel,
    config: dict[str, Any] | None = None,
) -> ExecutionPolicy:
    """Synthesize an execution policy from task model and config."""
    config = config or {}
    policy = ExecutionPolicy(
        parallelism=config.get("parallelism", 4),
        timeout_sec=config.get("timeout_sec", 600),
        max_sources=config.get("max_results_per_source", 20),
    )

    # Adjust for capabilities
    if "sandbox" in task_model.capabilities:
        policy.sandbox_enabled = True
        policy.timeout_sec = max(policy.timeout_sec, 1200)

    # Adjust for risks
    if len(task_model.dependencies) > 10:
        policy.parallelism = min(policy.parallelism, 2)
        policy.timeout_sec = max(policy.timeout_sec, 900)

    # Adjust for risk keywords
    for risk in task_model.risks:
        if "timeout" in risk.lower():
            policy.timeout_sec = max(policy.timeout_sec, 1200)
        if "rate" in risk.lower():
            policy.retry.max_retries = 3
            policy.retry.backoff_factor = 2.0

    logger.info(
        "Policy: parallelism=%d timeout=%ds sandbox=%s",
        policy.parallelism, policy.timeout_sec, policy.sandbox_enabled,
    )
    return policy
