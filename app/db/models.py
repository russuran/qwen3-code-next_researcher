from __future__ import annotations

import uuid

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    config = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    output_dir = Column(Text)
    error = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    started_at = Column(DateTime(timezone=True))
    finished_at = Column(DateTime(timezone=True))

    events = relationship("Event", back_populates="run", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="run", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_runs_status", "status"),
        Index("idx_runs_created", "created_at"),
    )


class Event(Base):
    __tablename__ = "events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    phase = Column(String(20), nullable=False)
    action = Column(String(100), nullable=False)
    tool_name = Column(String(100))
    tool_input = Column(JSONB)
    result_summary = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))

    run = relationship("Run", back_populates="events")

    __table_args__ = (
        Index("idx_events_run_id", "run_id"),
        Index("idx_events_run_created", "run_id", "created_at"),
    )


class Artifact(Base):
    __tablename__ = "artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    kind = Column(String(50), nullable=False)
    filename = Column(Text, nullable=False)
    uri = Column(Text, nullable=False)
    size_bytes = Column(BigInteger)
    content_type = Column(String(100), default="application/octet-stream")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))

    run = relationship("Run", back_populates="artifacts")

    __table_args__ = (
        Index("idx_artifacts_run_id", "run_id"),
    )
