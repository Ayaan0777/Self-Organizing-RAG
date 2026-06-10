from sqlalchemy import Column, Integer, Text, Float, Boolean, DateTime, String
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class QueryLog(Base):
    """One row per user query. Written by logger/query_logger.py."""
    __tablename__ = "autorag_query_log"
    id               = Column(Integer, primary_key=True)
    query            = Column(Text)
    chunk_ids        = Column(Text)        # JSON list of source metadata strings
    top_k_scores     = Column(Text)        # JSON list of floats e.g. [0.91, 0.84, ...]
    retrieved_chunks = Column(Text)        # JSON list of chunk text strings
    llm_response     = Column(Text)
    latency_ms       = Column(Integer)
    ctx_q_sim        = Column(Float)       # avg context↔question similarity
    answer_sem_sim   = Column(Float)       # answer↔ground_truth semantic similarity (null if no GT)
    flagged          = Column(Boolean, default=False)
    timestamp        = Column(DateTime, default=datetime.utcnow)


class LowRecallEvent(Base):
    """Fired by detector/detectors.py when one or more rules trigger."""
    __tablename__ = "autorag_low_recall_events"
    id                  = Column(Integer, primary_key=True)
    query_log_id        = Column(Integer)
    triggered_detectors = Column(Text)       # JSON list e.g. ["low_top_score", "llm_uncertainty"]
    severity            = Column(String(10)) # LOW | MEDIUM | HIGH
    resolved            = Column(Boolean, default=False)
    timestamp           = Column(DateTime, default=datetime.utcnow)
    attempts = Column(Integer, default=0)
    unfixable = Column(Boolean, default=False)


class RepairReport(Base):
    """Written by repair/orchestrator.py after each repair attempt."""
    __tablename__ = "autorag_repair_reports"
    id              = Column(Integer, primary_key=True)
    event_id        = Column(Integer)
    strategy_used   = Column(String(50))    # e.g. "increase_context", "reduce_noise"
    chunk_size_used = Column(Integer)       # the chunk_size param used for repair
    repair_reason   = Column(String(100))   # why this strategy was chosen
    chunks_before   = Column(Integer)
    chunks_after    = Column(Integer)
    score_before    = Column(Float)
    score_after     = Column(Float)
    resolved        = Column(Boolean, default=False)
    rolled_back     = Column(Boolean, default=False)  # True if repair was reverted
    duration_ms     = Column(Integer)
    timestamp       = Column(DateTime, default=datetime.utcnow)


class EvalSnapshot(Base):
    """Written by run_evaluation.py after each evaluation run."""
    __tablename__ = "autorag_eval_snapshots"
    id         = Column(Integer, primary_key=True)
    namespace  = Column(String(100))   # e.g. "mxbai-embed-large"
    llm        = Column(String(100))
    embeddings = Column(String(100))
    rouge_l    = Column(Float)
    sem_sim    = Column(Float)
    ctx_q_sim  = Column(Float)
    ctx_gt_sim = Column(Float)
    timestamp  = Column(DateTime, default=datetime.utcnow)
