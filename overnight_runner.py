#!/usr/bin/env python3
"""CLI entry point for the overnight research + implementation pipeline.

Examples
--------
# Standalone mode (generate implementations from scratch):
python overnight_runner.py "optimize QLoRA for Russian legal document classification" \
    --libraries "transformers,peft,datasets,torch,scikit-learn,faker"

# Repo mode (patch an existing repo, benchmark each hypothesis branch):
python overnight_runner.py "optimize QLoRA for Russian legal NLP" \
    --repo-url https://github.com/username/qlora-russian-legal \
    --libraries "transformers,peft,datasets,torch,scikit-learn"

# Point at uploaded dataset for real benchmarks:
python overnight_runner.py "..." --repo-url ... --data-dir uploads/datasets/my-dataset
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.logging import RichHandler

from core.llm import LLM, LLMConfig
from core.overnight_pipeline import OvernightPipeline

console = Console()


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@click.command()
@click.argument("topic")
@click.option(
    "--repo-url", default=None,
    help="GitHub/GitLab repo URL. Pipeline will clone it and generate hypotheses as patches.",
)
@click.option(
    "--libraries",
    default="transformers,peft,datasets,torch,accelerate,scikit-learn,faker,tokenizers",
    show_default=True,
    help="Comma-separated Python libraries the generated code may use.",
)
@click.option(
    "--workspace", default=None,
    help="Directory for all pipeline outputs (default: workspace/overnight-<run_id>).",
)
@click.option(
    "--sources", default="arxiv,github",
    show_default=True,
    help="Comma-separated research sources.",
)
@click.option(
    "--validation-model", default=None,
    help="HuggingFace model ID for hypothesis validation training. "
         "Defaults to cointegrated/rubert-tiny2 (83 MB Russian BERT).",
)
@click.option(
    "--max-iterations", default=2, show_default=True,
    help="Improvement iterations on the best hypothesis.",
)
@click.option(
    "--verbose", is_flag=True, help="Enable verbose logging.",
)
@click.option(
    "--config", "config_path", default="config.yaml", type=click.Path(),
    help="Path to config file.",
)
def main(
    topic: str,
    repo_url: str | None,
    libraries: str,
    workspace: str | None,
    sources: str,
    validation_model: str | None,
    max_iterations: int,
    verbose: bool,
    config_path: str,
) -> None:
    """Run the overnight research → implement → benchmark pipeline on TOPIC."""
    import os
    import uuid

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )
    for noisy in ("httpx", "litellm", "httpcore", "transformers", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Override validation model if specified
    if validation_model:
        os.environ["VALIDATION_MODEL"] = validation_model

    raw_cfg = _load_config(config_path)
    llm_section = raw_cfg.get("llm", {})
    llm_config = LLMConfig(**llm_section) if llm_section else LLMConfig()
    llm = LLM(llm_config)

    run_id = uuid.uuid4().hex[:8]
    ws = workspace or f"workspace/overnight-{run_id}"
    console.print(f"[bold]Overnight pipeline[/bold]  topic=[cyan]{topic}[/cyan]")
    if repo_url:
        console.print(f"  repo=[cyan]{repo_url}[/cyan]")
    console.print(f"  workspace=[dim]{ws}[/dim]")
    console.print(f"  validation model=[dim]{os.environ.get('VALIDATION_MODEL', 'cointegrated/rubert-tiny2')}[/dim]")

    pipeline = OvernightPipeline(
        llm=llm,
        workspace=ws,
        max_improvement_iterations=max_iterations,
    )

    async def _progress(phase: str, msg: str) -> None:
        console.print(f"  [bold blue]{phase}[/bold blue] {msg}")

    try:
        results = asyncio.run(
            pipeline.run(
                topic=topic,
                libraries=libraries,
                sources=sources.split(","),
                on_progress=_progress,
                repo_url=repo_url,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)

    # Summary
    final_metrics = results.get("final_metrics", {})
    acc = final_metrics.get("accuracy", final_metrics.get("eval_accuracy", "N/A"))
    f1  = final_metrics.get("f1", final_metrics.get("eval_f1", "N/A"))
    best = results.get("best_approach", {})
    best_title = best.get("hypothesis", {}).get("title", "N/A") if best else "N/A"

    console.print("\n[bold green]Done![/bold green]")
    console.print(f"  Status:       {results.get('status', 'unknown')}")
    console.print(f"  Best:         {best_title}")
    console.print(f"  Accuracy:     {acc}")
    console.print(f"  F1:           {f1}")
    console.print(f"  Report:       {results.get('report_path', ws + '/final_report.md')}")
    if repo_url:
        console.print(f"  Repo cloned:  {results.get('repo_context', {}).get('path', '')}")

    # Save summary
    summary_path = Path(ws) / "run_summary.json"
    summary_path.write_text(json.dumps({
        "topic": topic,
        "repo_url": repo_url,
        "status": results.get("status"),
        "best_hypothesis": best_title,
        "accuracy": acc,
        "f1": f1,
        "workspace": ws,
    }, indent=2, default=str))
    console.print(f"  Summary:      {summary_path}")


if __name__ == "__main__":
    main()
