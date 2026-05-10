"""
SQLAlchemy ORM models for the Auto-RAG diagnostic dashboard.

Tables:
  - QueryLog        : every query sent through the RAG pipeline
  - LowRecallEvent  : flagged queries that triggered detection rules
  - RepairReport    : results of repair attempts on flagged events
  - EvalSnapshot    : aggregate evaluation metrics per run
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    llm_response = Column(Text)
    top_k_scores = Column(Text)          # JSON list of floats
    retrieved_chunks = Column(Text)      # JSON list of strings
    flagged = Column(Boolean, default=False)
    latency_ms = Column(Float)
    ctx_q_sim = Column(Float, nullable=True)
    answer_sem_sim = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class LowRecallEvent(Base):
    __tablename__ = "low_recall_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_log_id = Column(Integer, ForeignKey("query_logs.id"), nullable=False)
    triggered_detectors = Column(Text)   # JSON list of rule names
    severity = Column(String(16))        # HIGH | MEDIUM | LOW
    resolved = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class RepairReport(Base):
    __tablename__ = "repair_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("low_recall_events.id"), nullable=False)
    strategy_used = Column(String(32))
    score_before = Column(Float, nullable=True)
    score_after = Column(Float, nullable=True)
    chunks_before = Column(Integer, nullable=True)
    chunks_after = Column(Integer, nullable=True)
    resolved = Column(Boolean, default=False)
    duration_ms = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class EvalSnapshot(Base):
    __tablename__ = "eval_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    namespace = Column(String(128))
    llm = Column(String(64))
    embeddings = Column(String(64))
    rouge_l = Column(Float)
    sem_sim = Column(Float)
    ctx_q_sim = Column(Float)
    ctx_gt_sim = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
