#!/usr/bin/env python3
"""CLI entry point for the autonomous research agent."""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.logging import RichHandler

from core.agent import AgentConfig, ResearchAgent
from core.llm import LLM, LLMConfig

console = Console()


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        console.print(f"[yellow]Config not found: {config_path}, using defaults[/yellow]")
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@click.command()
@click.argument("topic")
@click.option("--output-dir", default=None, help="Output directory for results")
@click.option("--sources", default=None, help="Comma-separated sources: arxiv,github,semantic_scholar,papers_with_code")
@click.option("--max-results", default=None, type=int, help="Max results per source")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--intervene", is_flag=True, help="Interactive mode with pauses between phases")
@click.option("--config", "config_path", default="config.yaml", type=click.Path(), help="Path to config file")
def main(
    topic: str,
    output_dir: str | None,
    sources: str | None,
    max_results: int | None,
    verbose: bool,
    intervene: bool,
    config_path: str,
) -> None:
    """Run an autonomous research investigation on TOPIC."""
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )
    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Load config
    raw_cfg = _load_config(config_path)

    # Build LLM config
    llm_section = raw_cfg.get("llm", {})
    llm_config = LLMConfig(**llm_section) if llm_section else LLMConfig()
    llm = LLM(llm_config)

    # Build agent config
    search_section = raw_cfg.get("search", {})
    output_section = raw_cfg.get("output", {})
    journal_section = raw_cfg.get("journal", {})

    agent_config = AgentConfig(
        output_dir=output_dir or output_section.get("base_dir", "./output"),
        journal_dir=journal_section.get("base_dir", "./journal"),
        sources=(
            sources.split(",") if sources
            else search_section.get("sources", ["arxiv", "semantic_scholar", "github", "papers_with_code"])
        ),
        max_results_per_source=max_results or search_section.get("max_results_per_source", 20),
        parallel_search=search_section.get("parallel", True),
        verbose=verbose,
        intervene=intervene,
    )

    agent = ResearchAgent(config=agent_config, llm=llm)

    # Run
    try:
        result_path = asyncio.run(agent.run(topic))
        console.print(f"\n[bold green]Done![/bold green] Results in: {result_path}")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
