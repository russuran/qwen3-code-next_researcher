"""Extend events and runs: add latency, cost, parent_event, run mode/autonomy

Revision ID: 002
Revises: 001
Create Date: 2026-04-05
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Extend events
    op.add_column("events", sa.Column("latency_ms", sa.BigInteger()))
    op.add_column("events", sa.Column("cost_estimate", sa.Float()))
    op.add_column("events", sa.Column("parent_event_id", sa.BigInteger()))

    # Extend runs
    op.add_column("runs", sa.Column("mode", sa.String(20), server_default="greenfield"))
    op.add_column("runs", sa.Column("task_type", sa.String(50)))
    op.add_column("runs", sa.Column("autonomy_level", sa.String(20), server_default="full"))


def downgrade() -> None:
    op.drop_column("events", "latency_ms")
    op.drop_column("events", "cost_estimate")
    op.drop_column("events", "parent_event_id")
    op.drop_column("runs", "mode")
    op.drop_column("runs", "task_type")
    op.drop_column("runs", "autonomy_level")
