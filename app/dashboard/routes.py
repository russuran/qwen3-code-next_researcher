"""Dashboard routes: Jinja2 + HTMX server-side rendered UI."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

import jinja2
from fastapi import APIRouter, Depends, Form, Request
import io
import zipfile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Run, Event
from app.deps import get_db, get_settings
from app.schemas.runs import RunCreate
from app.services import run_service
from app.settings import Settings

router = APIRouter()

_TEMPLATE_DIR = str(Path(__file__).parent / "templates")
_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_TEMPLATE_DIR),
    autoescape=True,
)
_env.filters["tojson"] = lambda v: json.dumps(v, default=str, ensure_ascii=False)
_env.tests["mapping"] = lambda v: isinstance(v, dict)


def _render(template_name: str, **ctx) -> HTMLResponse:
    tmpl = _env.get_template(template_name)
    return HTMLResponse(tmpl.render(**ctx))


@router.get("", response_class=HTMLResponse)
async def dashboard_index(request: Request):
    return _render("runs_list.html")


@router.get("/upload-dataset", response_class=HTMLResponse)
async def upload_dataset_page(request: Request):
    # List existing datasets
    datasets = []
    uploads_dir = Path("uploads/datasets")
    if uploads_dir.exists():
        for d in sorted(uploads_dir.iterdir(), reverse=True):
            if d.is_dir():
                samples = len(list(d.glob("*")))
                size_bytes = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                size = f"{size_bytes / 1024 / 1024:.1f} MB" if size_bytes > 1024*1024 else f"{size_bytes / 1024:.1f} KB"
                datasets.append({
                    "name": d.name,
                    "samples": samples,
                    "size": size,
                    "date": datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                })
    return _render("upload_dataset.html", datasets=datasets)


@router.post("/upload-dataset", response_class=HTMLResponse)
async def upload_dataset_submit(
    request: Request,
    dataset_name: str = Form(...),
):
    from datetime import datetime
    form = await request.form()
    files = form.getlist("files")

    if not files or not dataset_name.strip():
        return HTMLResponse('<div style="color: #ef4444;">No files or name provided</div>')

    safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in dataset_name.strip())
    dest = Path("uploads/datasets") / safe_name
    dest.mkdir(parents=True, exist_ok=True)

    count = 0
    for file in files:
        if not hasattr(file, 'filename') or not file.filename:
            continue
        content = await file.read()
        fname = file.filename

        if fname.endswith(".zip"):
            # Extract ZIP
            import zipfile as zf
            import io
            with zf.ZipFile(io.BytesIO(content)) as z:
                z.extractall(dest)
                count += len(z.namelist())
        else:
            (dest / fname).write_bytes(content)
            count += 1

    return HTMLResponse(
        f'<div style="padding: 0.5rem; background: #052e16; border: 1px solid #166534; border-radius: 0.375rem;">'
        f'Uploaded {count} files to <code>uploads/datasets/{safe_name}</code></div>'
    )


@router.get("/runs/table", response_class=HTMLResponse)
async def runs_table(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Run).order_by(Run.created_at.desc()).limit(50))
    runs = list(result.scalars().all())
    return _render("runs_table.html", runs=runs)


@router.post("/runs", response_class=HTMLResponse)
async def create_run_form(
    request: Request,
    topic: str = Form(...),
    extra_prompt: str = Form(""),
    max_results: int = Form(7),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    form_data = await request.form()
    sources = form_data.getlist("sources") or ["github", "arxiv"]
    stages = form_data.getlist("stages") or [
        "plan", "search", "filter", "deep_fetch", "analyze",
        "iterative", "hypotheses", "contradictions", "synthesize",
    ]
    max_iterations = int(form_data.get("max_iterations", 3))
    is_overnight = "overnight" in form_data
    libraries = form_data.get("libraries", "pytesseract, easyocr, Pillow, opencv-python")

    # Handle context file
    context_text = ""
    context_file = form_data.get("context_file")
    if context_file and hasattr(context_file, 'filename') and context_file.filename:
        content = await context_file.read()
        context_text = content.decode("utf-8", errors="ignore")[:10000]

    # Build enriched topic
    enriched_topic = topic
    if extra_prompt.strip():
        enriched_topic += f"\n\nAdditional instructions: {extra_prompt.strip()}"
    if context_text:
        enriched_topic += f"\n\nContext from file ({context_file.filename}):\n{context_text[:5000]}"

    # Handle dataset upload
    dataset_file = form_data.get("dataset_file")
    dataset_url = form_data.get("dataset_url", "").strip()

    if is_overnight:
        # Launch overnight pipeline
        import httpx as httpx_client
        try:
            async with httpx_client.AsyncClient() as client:
                resp = await client.post(
                    "http://localhost:8000/overnight",
                    json={"topic": enriched_topic, "libraries": libraries, "max_iterations": max_iterations, "sources": list(sources)},
                    timeout=30,
                )
                if resp.status_code == 201:
                    data = resp.json()
                    run_id = data.get("run_id", "")

                    # Upload dataset to overnight workspace if provided
                    if dataset_file and hasattr(dataset_file, 'filename') and dataset_file.filename:
                        dest = Path(f"workspace/overnight-{run_id[:8]}") / "user_data"
                        dest.mkdir(parents=True, exist_ok=True)
                        file_content = await dataset_file.read()
                        if dataset_file.filename.endswith(".zip"):
                            import zipfile as zf_mod
                            import io as io_mod
                            with zf_mod.ZipFile(io_mod.BytesIO(file_content)) as z:
                                z.extractall(dest)
                        else:
                            (dest / dataset_file.filename).write_bytes(file_content)

                    result = await db.execute(select(Run).order_by(Run.created_at.desc()).limit(50))
                    runs = list(result.scalars().all())
                    return _render("runs_table.html", runs=runs)
        except Exception:
            pass

    # Regular research run
    body = RunCreate(
        topic=enriched_topic,
        sources=list(sources),
        max_results_per_source=max_results,
    )
    run = await run_service.create_run(db, body)

    if dataset_file and hasattr(dataset_file, 'filename') and dataset_file.filename:
        dest = Path(f"output/run-{str(run.id)[:8]}") / "user_data"
        dest.mkdir(parents=True, exist_ok=True)
        file_content = await dataset_file.read()
        if dataset_file.filename.endswith(".zip"):
            import zipfile as zf_mod
            import io as io_mod
            with zf_mod.ZipFile(io_mod.BytesIO(file_content)) as z:
                z.extractall(dest)
        else:
            (dest / dataset_file.filename).write_bytes(file_content)

    run.config = {
        **(run.config or {}),
        "stages": list(stages),
        "extra_prompt": extra_prompt.strip(),
        "original_topic": topic,
        "max_iterations": max_iterations,
        "dataset_url": dataset_url,
    }
    await db.commit()

    run_service.start_run_background(run.id, body, settings, stages=list(stages))

    # Return updated table
    result = await db.execute(select(Run).order_by(Run.created_at.desc()).limit(50))
    runs = list(result.scalars().all())
    return _render("runs_table.html", runs=runs)


@router.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: str, db: AsyncSession = Depends(get_db)):
    try:
        uid = UUID(run_id)
    except ValueError:
        return HTMLResponse("Invalid run ID", status_code=400)

    run = await db.get(Run, uid)
    if not run:
        return HTMLResponse("Run not found", status_code=404)

    events_result = await db.execute(
        select(Event).where(Event.run_id == uid).order_by(Event.created_at)
    )
    events = list(events_result.scalars().all())

    # Load evaluation metrics if available
    metrics = None
    artifacts = []
    if run.output_dir:
        eval_path = Path(run.output_dir) / "09_evaluation.json"
        if eval_path.exists():
            try:
                metrics = json.loads(eval_path.read_text())
            except Exception:
                pass

        # Collect downloadable artifacts
        out_path = Path(run.output_dir)
        if out_path.exists():
            _ARTIFACT_INFO = {
                "01_plan.json": ("Research Plan", "plan"),
                "02_sources.json": ("Raw Sources", "search"),
                "06_comparison.md": ("Comparison Table", "analyze"),
                "07_synthesis.md": ("Final Report", "synthesize"),
                "08_references.md": ("References", "synthesize"),
                "09_evaluation.json": ("Quality Metrics", "synthesize"),
            }
            for fname, (label, phase) in _ARTIFACT_INFO.items():
                fpath = out_path / fname
                if fpath.exists() and fpath.stat().st_size > 0:
                    artifacts.append({
                        "filename": fname,
                        "label": label,
                        "phase": phase,
                        "size": fpath.stat().st_size,
                        "download_url": f"/dashboard/runs/{run_id}/download/{fname}",
                    })
            # Summaries dir
            summaries_dir = out_path / "05_summaries"
            if summaries_dir.exists():
                count = len(list(summaries_dir.glob("*.json")))
                if count > 0:
                    artifacts.append({
                        "filename": "05_summaries/",
                        "label": f"Analysis Summaries ({count} files)",
                        "phase": "analyze",
                        "size": sum(f.stat().st_size for f in summaries_dir.glob("*.json")),
                        "download_url": f"/dashboard/runs/{run_id}/download-summaries",
                    })

    # Group events by phase for phase logs
    phase_events: dict[str, list] = {}
    for e in events:
        phase_events.setdefault(e.phase, []).append(e)

    # Load hypotheses data — check multiple locations
    hypotheses_data = None
    search_paths = []
    if run.output_dir:
        search_paths.append(Path(run.output_dir) / "04_hypotheses.json")
    # Also check workspace for overnight runs
    run_id_short = str(run.id)[:8]
    for ws in Path("workspace").glob(f"*{run_id_short}*"):
        search_paths.append(ws / "overnight_results.json")
        for sub in ws.glob("research/*/04_hypotheses.json"):
            search_paths.append(sub)

    for hyp_path in search_paths:
        if hyp_path.exists():
            try:
                data = json.loads(hyp_path.read_text(encoding="utf-8"))
                # overnight_results has hypotheses inside implementations
                if "implementations" in data:
                    impls = data.get("implementations", [])
                    hypotheses_data = {
                        "hypotheses": [
                            {**impl.get("hypothesis", {}),
                             "status": "validated" if impl.get("smoke_test_passed") else "rejected",
                             "benchmark": impl.get("benchmark_metrics", {})}
                            for impl in impls
                        ],
                        "consensus": data.get("research", {}).get("consensus", []),
                        "gaps": data.get("research", {}).get("gaps", []),
                    }
                    break
                elif "hypotheses" in data:
                    hypotheses_data = data
                    break
            except Exception:
                continue

    # Check if data is available for benchmark
    data_available = False
    if run.output_dir:
        user_data = Path(run.output_dir) / "user_data"
        data_available = user_data.exists() and any(user_data.rglob("*.*"))
    if not data_available:
        data_available = Path("uploads/datasets").exists() and any(Path("uploads/datasets").rglob("*.*"))

    # Load auto-generated labels for validation
    labels = []
    data_dir = None
    if data_available:
        for check_dir in [Path("uploads/datasets")]:
            if check_dir.exists():
                for d in sorted(check_dir.iterdir(), reverse=True):
                    if d.is_dir() and any(d.rglob("*.*")):
                        data_dir = d
                        break
                if data_dir:
                    break
    if data_dir:
        for gt_file in sorted(data_dir.rglob("ground_truth.json"))[:30]:
            try:
                gt = json.loads(gt_file.read_text(encoding="utf-8"))
                source_file = gt.get("source_file", str(gt_file.relative_to(data_dir)))
                # Find image next to ground_truth
                img_file = None
                for ext in [".jpg", ".jpeg", ".png"]:
                    candidate = gt_file.parent / (gt_file.stem.replace("ground_truth", "") + ext).lstrip(".")
                    if not candidate.exists():
                        for img in gt_file.parent.glob(f"*{ext}"):
                            candidate = img
                            break
                    if candidate.exists():
                        img_file = candidate
                        break

                labels.append({
                    "gt_path": str(gt_file),
                    "source_file": source_file[:50],
                    "filename": gt_file.parent.name,
                    "fields": gt.get("fields", {}),
                    "raw_text": gt.get("raw_text", ""),
                    "validated": gt.get("validated", False),
                    "image_url": f"/dashboard/serve-image?path={img_file}" if img_file else None,
                })
            except Exception:
                continue

    return _render(
        "run_detail.html",
        run=run, events=events, metrics=metrics,
        artifacts=artifacts, phase_events=phase_events,
        hypotheses_data=hypotheses_data, data_available=data_available,
        labels=labels,
    )


@router.get("/serve-image")
async def serve_image(path: str):
    """Serve dataset images for label validation."""
    from fastapi.responses import FileResponse as FR
    p = Path(path)
    if not p.exists() or not p.is_file():
        return HTMLResponse("Not found", status_code=404)
    content_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(p.suffix.lower().lstrip("."), "application/octet-stream")
    return FR(str(p), media_type=content_type)


@router.post("/runs/{run_id}/validate-label", response_class=HTMLResponse)
async def validate_label(request: Request, run_id: str):
    """Save validated/corrected label."""
    form = await request.form()
    gt_path = form.get("label_path", "")
    if not gt_path or not Path(gt_path).exists():
        return HTMLResponse('<span class="text-xs" style="color: var(--red-text);">File not found</span>')

    try:
        gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
        # Update fields from form
        fields = {}
        for key, value in form.items():
            if key.startswith("field_") and key != "field_raw_text":
                field_name = key[6:]
                fields[field_name] = value
        gt["fields"] = fields
        # Update raw_text if provided
        raw_text = form.get("field_raw_text", "")
        if raw_text:
            gt["raw_text"] = raw_text
        gt["validated"] = True
        gt["validated_by"] = "human"
        Path(gt_path).write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
        return HTMLResponse('<span class="text-xs" style="color: var(--green-text);">Saved</span>')
    except Exception as e:
        return HTMLResponse(f'<span class="text-xs" style="color: var(--red-text);">{e}</span>')


@router.post("/runs/{run_id}/rerun-benchmark", response_class=HTMLResponse)
async def rerun_benchmark(
    request: Request, run_id: str,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Re-run benchmark on validated labels."""
    import asyncio as aio

    try:
        uid = UUID(run_id)
    except ValueError:
        return HTMLResponse("Invalid ID", status_code=400)

    run = await db.get(Run, uid)
    if not run:
        return HTMLResponse("Run not found", status_code=404)

    # Find workspace with implementations
    run_short = str(run.id)[:8]
    workspace = None
    for ws in Path("workspace").glob(f"*{run_short}*"):
        if ws.is_dir():
            workspace = ws
            break

    if not workspace:
        return HTMLResponse('<span style="color: var(--red-text);">No workspace found for this run</span>')

    # Find implementations
    impl_dirs = [d for d in workspace.iterdir() if d.is_dir() and (d / "implementation.py").exists()]
    if not impl_dirs:
        return HTMLResponse('<span style="color: var(--red-text);">No implementations found</span>')

    # Launch benchmark in background
    async def _do_benchmark():
        from core.overnight_pipeline import OvernightPipeline
        from app.deps import build_llm
        llm = build_llm(settings)
        pipeline = OvernightPipeline(llm=llm, workspace=str(workspace))

        results = []
        for impl_dir in impl_dirs:
            metrics = await pipeline._run_benchmark(impl_dir)
            results.append({"dir": impl_dir.name, "metrics": metrics})

            # Log event
            if db_session.async_session_factory:
                async with db_session.async_session_factory() as session:
                    event = Event(
                        run_id=uid, phase="benchmark", action="rerun_benchmark",
                        result_summary=f"{impl_dir.name}: {json.dumps(metrics)[:100]}",
                    )
                    session.add(event)
                    await session.commit()

        # Save results
        (workspace / "rerun_benchmark_results.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )

    aio.create_task(_do_benchmark())
    return HTMLResponse(
        f'<div style="padding: 0.4rem; background: #1e3a5f; border: 1px solid var(--accent); border-radius: 0.25rem;">'
        f'Re-running benchmark on {len(impl_dirs)} implementations with validated labels. Check events for progress.'
        f'</div>'
    )


@router.get("/runs/{run_id}/events-partial", response_class=HTMLResponse)
async def events_partial(request: Request, run_id: str, db: AsyncSession = Depends(get_db)):
    try:
        uid = UUID(run_id)
    except ValueError:
        return HTMLResponse("Invalid", status_code=400)

    events_result = await db.execute(
        select(Event).where(Event.run_id == uid).order_by(Event.created_at)
    )
    events = list(events_result.scalars().all())

    phase_colors = {"plan": "#a78bfa", "search": "#38bdf8", "analyze": "#fb923c", "synthesize": "#22c55e"}
    html_parts = []
    for e in events:
        color = phase_colors.get(e.phase, "#94a3b8")
        tool_html = f'<div class="event-tool"><span class="tool-icon">&gt;</span> {e.tool_name}</div>' if e.tool_name else ""
        summary_html = f'<div class="event-summary">{(e.result_summary or "")[:150]}</div>' if e.result_summary else ""
        time_str = e.created_at.strftime('%H:%M:%S') if e.created_at else ""
        html_parts.append(
            f'<div class="event-item" data-phase="{e.phase}">'
            f'<div class="event-header">'
            f'<span class="event-phase" style="color:{color};font-weight:700;font-size:0.8rem;">[{e.phase}]</span> '
            f'<span class="event-action" style="color:#e2e8f0;font-size:0.85rem;">{e.action}</span>'
            f'<span style="margin-left:auto;font-size:0.75rem;color:#475569;font-family:monospace;">{time_str}</span>'
            f'</div>'
            f'{tool_html}{summary_html}'
            f'</div>'
        )

    return HTMLResponse("\n".join(html_parts) if html_parts else '<div style="color: #64748b; padding: 1rem;">Waiting for events...</div>')


@router.get("/runs/{run_id}/current-activity", response_class=HTMLResponse)
async def current_activity(request: Request, run_id: str, db: AsyncSession = Depends(get_db)):
    """Returns the latest event as a live activity indicator."""
    try:
        uid = UUID(run_id)
    except ValueError:
        return HTMLResponse("")

    run = await db.get(Run, uid)
    if not run or run.status in ("completed", "failed", "cancelled"):
        if run and run.status == "completed":
            return HTMLResponse(
                '<span style="color: #22c55e; font-weight: 600;">Research complete</span>'
            )
        elif run and run.status == "failed":
            return HTMLResponse(
                f'<span style="color: #ef4444; font-weight: 600;">Failed: {(run.error or "unknown")[:100]}</span>'
            )
        return HTMLResponse("")

    events_result = await db.execute(
        select(Event).where(Event.run_id == uid).order_by(Event.created_at.desc()).limit(1)
    )
    last_event = events_result.scalar_one_or_none()

    if not last_event:
        return HTMLResponse(
            '<div class="activity-pulse"></div>'
            '<span style="color: #94a3b8;">Initializing...</span>'
        )

    phase_labels = {"plan": "Planning", "search": "Searching", "analyze": "Analyzing", "synthesize": "Synthesizing"}
    phase_colors = {"plan": "#a78bfa", "search": "#38bdf8", "analyze": "#fb923c", "synthesize": "#22c55e"}
    phase = last_event.phase
    label = phase_labels.get(phase, phase)
    color = phase_colors.get(phase, "#94a3b8")

    detail = ""
    if last_event.tool_name:
        detail += f' <span style="color: #a78bfa;">&gt; {last_event.tool_name}</span>'
    if last_event.result_summary:
        summary = last_event.result_summary[:120]
        detail += f' <span style="color: #cbd5e1;">— {summary}</span>'

    return HTMLResponse(
        f'<div class="activity-pulse" style="background: {color};"></div>'
        f'<span style="color: {color}; font-weight: 600;">{label}</span>'
        f'{detail}'
    )


@router.get("/runs/{run_id}/status-badge", response_class=HTMLResponse)
async def status_badge(request: Request, run_id: str, db: AsyncSession = Depends(get_db)):
    try:
        uid = UUID(run_id)
    except ValueError:
        return HTMLResponse("")

    run = await db.get(Run, uid)
    if not run:
        return HTMLResponse("")

    # Determine real phase from latest event
    events_result = await db.execute(
        select(Event).where(Event.run_id == uid).order_by(Event.created_at.desc()).limit(1)
    )
    last_event = events_result.scalar_one_or_none()
    effective_status = run.status
    if last_event and run.status not in ("completed", "failed", "cancelled"):
        phase_map = {"plan": "planning", "search": "searching", "analyze": "analyzing", "synthesize": "synthesizing"}
        effective_status = phase_map.get(last_event.phase, run.status)

    extra = ""
    if effective_status in ("completed", "failed", "cancelled"):
        extra = ""  # stop polling
    else:
        extra = f'hx-get="/dashboard/runs/{run_id}/status-badge" hx-trigger="every 3s" hx-swap="outerHTML"'

    return HTMLResponse(
        f'<span class="badge badge-{effective_status}" id="run-status" {extra}>{effective_status}</span>'
    )


@router.post("/runs/{run_id}/cancel", response_class=HTMLResponse)
async def cancel_run_dashboard(request: Request, run_id: str, db: AsyncSession = Depends(get_db)):
    try:
        uid = UUID(run_id)
    except ValueError:
        return HTMLResponse("Invalid", status_code=400)

    run = await db.get(Run, uid)
    if run and run.status not in ("completed", "failed", "cancelled"):
        run.status = "cancelled"
        await db.commit()

    result = await db.execute(select(Run).order_by(Run.created_at.desc()).limit(50))
    runs = list(result.scalars().all())
    return _render("runs_table.html", runs=runs)


@router.post("/runs/{run_id}/upload-data", response_class=HTMLResponse)
async def upload_run_data(request: Request, run_id: str, db: AsyncSession = Depends(get_db)):
    """Upload data files directly to a run's workspace."""
    try:
        uid = UUID(run_id)
    except ValueError:
        return HTMLResponse("Invalid ID", status_code=400)

    run = await db.get(Run, uid)
    if not run:
        return HTMLResponse("Run not found", status_code=404)

    form = await request.form()
    files = form.getlist("files")

    # Save to run's output dir or workspace
    dest = Path(run.output_dir or f"workspace/run-{run_id[:8]}") / "user_data"
    dest.mkdir(parents=True, exist_ok=True)

    count = 0
    for file in files:
        if not hasattr(file, 'filename') or not file.filename:
            continue
        content = await file.read()
        fname = file.filename

        if fname.endswith(".zip"):
            import zipfile as zf
            import io
            with zf.ZipFile(io.BytesIO(content)) as z:
                z.extractall(dest)
                count += len(z.namelist())
        else:
            (dest / fname).write_bytes(content)
            count += 1

    return HTMLResponse(
        f'<div style="padding: 0.5rem; background: #052e16; border: 1px solid var(--green); border-radius: 0.375rem;" class="text-sm">'
        f'Uploaded {count} files to <code>{dest}</code></div>'
    )


@router.post("/runs/{run_id}/execute-hypotheses", response_class=HTMLResponse)
async def execute_hypotheses_dashboard(
    request: Request, run_id: str,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Trigger hypothesis implementation loop from dashboard."""
    import httpx as httpx_client
    try:
        async with httpx_client.AsyncClient() as client:
            resp = await client.post(
                f"http://localhost:8000/hypotheses/{run_id}/execute",
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                impl_id = data.get("implementation_run_id", "")
                count = data.get("hypotheses_count", 0)
                return HTMLResponse(
                    f'<div style="padding: 0.5rem; background: #1e3a5f; border: 1px solid #2563eb; border-radius: 0.375rem;">'
                    f'Started implementation of {count} hypotheses. '
                    f'<a href="/dashboard/runs/{impl_id}" style="color: #38bdf8;">View progress</a>'
                    f'</div>'
                )
            else:
                detail = resp.json().get("detail", resp.text[:200]) if resp.headers.get("content-type", "").startswith("application/json") else resp.text[:200]
                return HTMLResponse(
                    f'<div style="padding: 0.5rem; background: #450a0a; border: 1px solid #991b1b; border-radius: 0.375rem;">{detail}</div>'
                )
    except Exception as e:
        return HTMLResponse(
            f'<div style="padding: 0.5rem; background: #450a0a; border: 1px solid #991b1b; border-radius: 0.375rem;">Error: {str(e)[:200]}</div>'
        )


# ---------------------------------------------------------------------------
# File viewer
# ---------------------------------------------------------------------------

_FILE_INFO = {
    "01_plan.json": "Research Plan",
    "02_sources.json": "Raw Sources",
    "06_comparison.md": "Comparison Table",
    "07_synthesis.md": "Final Report",
    "08_references.md": "References",
    "09_evaluation.json": "Quality Metrics",
}


def _md_to_html(md_text: str) -> str:
    """Minimal markdown to HTML converter."""
    import re
    html = md_text

    # Code blocks
    def _code_block(m):
        lang = m.group(1) or ""
        code = m.group(2).replace("<", "&lt;").replace(">", "&gt;")
        return f'<pre><code>{code}</code></pre>'
    html = re.sub(r'```(\w*)\n(.*?)```', _code_block, html, flags=re.DOTALL)

    # Inline code
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)

    # Tables
    lines = html.split("\n")
    in_table = False
    table_lines = []
    result_lines = []
    for line in lines:
        stripped = line.strip()
        if "|" in stripped and stripped.startswith("|"):
            if re.match(r'^\|[\s\-:|]+\|$', stripped):
                continue  # separator row
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not in_table:
                in_table = True
                result_lines.append("<table>")
                result_lines.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr></thead><tbody>")
            else:
                result_lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        else:
            if in_table:
                in_table = False
                result_lines.append("</tbody></table>")
            result_lines.append(line)
    if in_table:
        result_lines.append("</tbody></table>")
    html = "\n".join(result_lines)

    # Headers
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

    # Bold, italic
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

    # Links
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', html)

    # Lists
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

    # Horizontal rules
    html = re.sub(r'^---+$', r'<hr>', html, flags=re.MULTILINE)

    # Paragraphs (double newline)
    html = re.sub(r'\n\n', r'</p><p>', html)
    html = f'<p>{html}</p>'
    html = html.replace('<p></p>', '')
    html = html.replace('<p><h', '<h').replace('</h1></p>', '</h1>')
    html = html.replace('</h2></p>', '</h2>').replace('</h3></p>', '</h3>')
    html = html.replace('<p><pre>', '<pre>').replace('</pre></p>', '</pre>')
    html = html.replace('<p><table>', '<table>').replace('</table></p>', '</table>')
    html = html.replace('<p><hr></p>', '<hr>')

    return html


@router.get("/runs/{run_id}/view/{filename:path}", response_class=HTMLResponse)
async def view_file(request: Request, run_id: str, filename: str, db: AsyncSession = Depends(get_db)):
    try:
        uid = UUID(run_id)
    except ValueError:
        return HTMLResponse("Invalid run ID", status_code=400)

    run = await db.get(Run, uid)
    if not run or not run.output_dir:
        return HTMLResponse("Not found", status_code=404)

    file_path = Path(run.output_dir) / filename
    if not file_path.exists() or not file_path.is_file():
        return HTMLResponse("File not found", status_code=404)

    content = file_path.read_text(encoding="utf-8", errors="ignore")
    ext = file_path.suffix.lower()

    if ext == ".md":
        file_type = "markdown"
        rendered_html = _md_to_html(content)
    elif ext == ".json":
        file_type = "json"
        try:
            content = json.dumps(json.loads(content), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        rendered_html = ""
    else:
        file_type = "text"
        rendered_html = ""

    size = file_path.stat().st_size
    size_display = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"

    # Build sibling file list
    siblings = []
    out_path = Path(run.output_dir)
    for fname, label in _FILE_INFO.items():
        if (out_path / fname).exists():
            siblings.append({"filename": fname, "label": label})

    return _render(
        "file_viewer.html",
        run_id=str(run.id), filename=filename, content=content,
        file_type=file_type, rendered_html=rendered_html,
        size_display=size_display, siblings=siblings,
    )


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/download/{filename:path}")
async def download_artifact(run_id: str, filename: str, db: AsyncSession = Depends(get_db)):
    try:
        uid = UUID(run_id)
    except ValueError:
        return HTMLResponse("Invalid", status_code=400)

    run = await db.get(Run, uid)
    if not run or not run.output_dir:
        return HTMLResponse("Not found", status_code=404)

    file_path = Path(run.output_dir) / filename
    if not file_path.exists() or not file_path.is_file():
        return HTMLResponse("File not found", status_code=404)

    return FileResponse(
        str(file_path),
        filename=filename,
        media_type="application/octet-stream",
    )


@router.get("/runs/{run_id}/download-summaries")
async def download_summaries(run_id: str, db: AsyncSession = Depends(get_db)):
    """Download all analysis summaries as a ZIP."""
    try:
        uid = UUID(run_id)
    except ValueError:
        return HTMLResponse("Invalid", status_code=400)

    run = await db.get(Run, uid)
    if not run or not run.output_dir:
        return HTMLResponse("Not found", status_code=404)

    summaries_dir = Path(run.output_dir) / "05_summaries"
    if not summaries_dir.exists():
        return HTMLResponse("No summaries", status_code=404)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(summaries_dir.glob("*.json")):
            zf.write(f, f.name)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=summaries_{run_id[:8]}.zip"},
    )


@router.get("/runs/{run_id}/download-all")
async def download_all(run_id: str, db: AsyncSession = Depends(get_db)):
    """Download entire output directory as a ZIP."""
    try:
        uid = UUID(run_id)
    except ValueError:
        return HTMLResponse("Invalid", status_code=400)

    run = await db.get(Run, uid)
    if not run or not run.output_dir:
        return HTMLResponse("Not found", status_code=404)

    out_path = Path(run.output_dir)
    if not out_path.exists():
        return HTMLResponse("Output not found", status_code=404)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out_path.rglob("*")):
            if f.is_file():
                zf.write(f, str(f.relative_to(out_path)))
    buf.seek(0)

    slug = out_path.name
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={slug}.zip"},
    )


# ---------------------------------------------------------------------------
# Phase logs
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/phase-log/{phase}", response_class=HTMLResponse)
async def phase_log(request: Request, run_id: str, phase: str, db: AsyncSession = Depends(get_db)):
    try:
        uid = UUID(run_id)
    except ValueError:
        return HTMLResponse("Invalid", status_code=400)

    events_result = await db.execute(
        select(Event).where(Event.run_id == uid, Event.phase == phase).order_by(Event.created_at)
    )
    events = list(events_result.scalars().all())

    phase_colors = {"plan": "#a78bfa", "search": "#38bdf8", "analyze": "#fb923c", "synthesize": "#22c55e"}
    color = phase_colors.get(phase, "#94a3b8")

    html_parts = [f'<div style="padding: 0.5rem 0; border-bottom: 1px solid #334155; color: {color}; font-weight: 700;">{phase.upper()} — {len(events)} events</div>']
    for e in events:
        tool_html = f'<span style="color: #a78bfa;">&gt; {e.tool_name}</span> ' if e.tool_name else ""
        time_str = e.created_at.strftime('%H:%M:%S') if e.created_at else ""
        summary = (e.result_summary or "")[:200]
        html_parts.append(
            f'<div style="padding: 0.4rem 0; border-bottom: 1px solid #1e293b; font-size: 0.85rem;">'
            f'<span style="color: #64748b; font-family: monospace; font-size: 0.75rem;">{time_str}</span> '
            f'<span style="color: #e2e8f0;">{e.action}</span> '
            f'{tool_html}'
            f'<div style="color: #94a3b8; font-size: 0.8rem; padding-left: 1rem; margin-top: 0.15rem;">{summary}</div>'
            f'</div>'
        )

    if not events:
        html_parts.append('<div style="color: #475569; padding: 1rem;">No events in this phase.</div>')

    return HTMLResponse("\n".join(html_parts))
