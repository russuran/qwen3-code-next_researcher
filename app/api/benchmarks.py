"""Benchmarks API."""
from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Run
from app.deps import get_db

router = APIRouter()


class BenchmarkRequest(BaseModel):
    run_id: str
    benchmark_set: str = "validation"


@router.post("/run", status_code=201)
async def run_benchmark(body: BenchmarkRequest, db: AsyncSession = Depends(get_db)):
    try:
        uid = UUID(body.run_id)
    except ValueError:
        raise HTTPException(400, "Invalid run ID")
    run = await db.get(Run, uid)
    if not run:
        raise HTTPException(404, "Run not found")
    return {"status": "benchmark_queued", "run_id": body.run_id, "benchmark_set": body.benchmark_set}


@router.get("/{benchmark_id}")
async def get_benchmark(benchmark_id: str):
    return {"benchmark_id": benchmark_id, "status": "not_implemented_yet"}
