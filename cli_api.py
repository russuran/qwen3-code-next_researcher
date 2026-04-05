#!/usr/bin/env python3
"""Thin CLI client: sends requests to the Researcher API."""
from __future__ import annotations

import sys
import time

import click
import httpx
from rich.console import Console

console = Console()
DEFAULT_API = "http://localhost:8000"


@click.command()
@click.argument("topic")
@click.option("--api-url", default=DEFAULT_API, envvar="RESEARCHER_API_URL", help="API base URL")
@click.option("--sources", default=None, help="Comma-separated sources: arxiv,github,semantic_scholar,papers_with_code")
@click.option("--max-results", default=None, type=int, help="Max results per source")
@click.option("--verbose", is_flag=True, help="Enable verbose output")
@click.option("--poll-interval", default=5, type=int, help="Seconds between status polls")
def main(topic: str, api_url: str, sources: str | None, max_results: int | None, verbose: bool, poll_interval: int):
    """Submit a research run to the API and poll for completion."""
    payload = {"topic": topic, "verbose": verbose}
    if sources:
        payload["sources"] = sources.split(",")
    if max_results:
        payload["max_results_per_source"] = max_results

    # Create run
    try:
        resp = httpx.post(f"{api_url}/runs", json=payload, timeout=30)
        resp.raise_for_status()
    except httpx.ConnectError:
        console.print(f"[bold red]Cannot connect to API at {api_url}[/bold red]")
        console.print("Make sure the server is running: uvicorn app.main:app")
        sys.exit(1)

    run = resp.json()
    run_id = run["id"]
    console.print(f"[green]Run created:[/green] {run_id}")

    # Poll for completion
    last_event_count = 0
    while True:
        time.sleep(poll_interval)

        try:
            resp = httpx.get(f"{api_url}/runs/{run_id}", timeout=30)
            data = resp.json()
        except Exception as e:
            console.print(f"[yellow]Poll error: {e}[/yellow]")
            continue

        status = data["status"]

        # Fetch and display new events
        if verbose:
            try:
                events_resp = httpx.get(f"{api_url}/runs/{run_id}/events", timeout=30)
                events = events_resp.json()
                for ev in events[last_event_count:]:
                    console.print(f"  [dim][{ev['phase']}][/dim] {ev['action']}: {ev.get('result_summary', '')}")
                last_event_count = len(events)
            except Exception:
                pass

        if status in ("completed", "failed", "cancelled"):
            if status == "completed":
                console.print(f"\n[bold green]Done![/bold green] Output: {data.get('output_dir')}")
            elif status == "failed":
                console.print(f"\n[bold red]Failed:[/bold red] {data.get('error')}")
            else:
                console.print(f"\n[yellow]Cancelled[/yellow]")
            break
        else:
            if not verbose:
                console.print(f"  [dim]Status: {status}...[/dim]")


if __name__ == "__main__":
    main()
