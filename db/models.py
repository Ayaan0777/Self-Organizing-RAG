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
    # ── Stage 2 metrics (added for MEASURE loop) ──
    retrieval_precision  = Column(Float, nullable=True)    # precision@K: fraction of relevant chunks
    context_sufficiency  = Column(Boolean, nullable=True)  # True if context covers the full answer
    hallucination_rate   = Column(Float, nullable=True)    # 0.0–1.0: fraction of ungrounded claims
    question_category    = Column(String(30), nullable=True)  # short_factual | complex | cross_section


class LowRecallEvent(Base):
    """Fired by detector/detectors.py when one or more rules trigger."""
    __tablename__ = "autorag_low_recall_events"
    id                  = Column(Integer, primary_key=True)
    query_log_id        = Column(Integer)
    triggered_detectors = Column(Text)       # JSON list e.g. ["low_top_score", "llm_uncertainty"]
    severity            = Column(String(10)) # LOW | MEDIUM | HIGH
    resolved            = Column(Boolean, default=False)
    timestamp           = Column(DateTime, default=datetime.utcnow)
    attempts            = Column(Integer, default=0)
    unfixable           = Column(Boolean, default=False)
    # ── Stage 3 cooldown tracking ──
    last_repair_at      = Column(DateTime, nullable=True)
    cooldown_until      = Column(DateTime, nullable=True)
    source_document     = Column(String(200), nullable=True)


class RepairReport(Base):
    """Written by repair/orchestrator.py after each repair attempt."""
    __tablename__ = "autorag_repair_reports"
    id            = Column(Integer, primary_key=True)
    event_id      = Column(Integer)
    strategy_used = Column(String(50))   # semantic | llm | entropy
    chunks_before = Column(Integer)
    chunks_after  = Column(Integer)
    score_before  = Column(Float)
    score_after   = Column(Float)
    resolved      = Column(Boolean, default=False)
    original_answer = Column(Text, nullable=True)
    resolved_answer = Column(Text, nullable=True)
    # ── Enhanced repair metrics (precision, recall, accuracy) ──
    precision_before = Column(Float, nullable=True)
    precision_after  = Column(Float, nullable=True)
    recall_before    = Column(Float, nullable=True)
    recall_after     = Column(Float, nullable=True)
    accuracy_before  = Column(Float, nullable=True)
    accuracy_after   = Column(Float, nullable=True)
    duration_ms   = Column(Integer)
    timestamp     = Column(DateTime, default=datetime.utcnow)


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
    # ── Stage 2 new aggregate metrics ──
    retrieval_precision = Column(Float, nullable=True)
    context_sufficiency = Column(Float, nullable=True)  # stored as fraction (0.0–1.0)
    hallucination_rate  = Column(Float, nullable=True)
    timestamp  = Column(DateTime, default=datetime.utcnow)


# ══════════════════════════════════════════════════════════════
#  NEW TABLES — Stages 2-4 (MEASURE → DECIDE → ACT)
# ══════════════════════════════════════════════════════════════

class PipelineConfig(Base):
    """
    Tracks the current chunking configuration per namespace.
    Only one row should be active=True per namespace at a time.
    Enables rollback by preserving previous configurations.
    """
    __tablename__ = "autorag_pipeline_config"
    id              = Column(Integer, primary_key=True)
    namespace       = Column(String(100))
    chunk_size      = Column(Integer, default=250)
    chunk_overlap   = Column(Integer, default=80)
    chunk_strategy  = Column(String(50), default="semantic")
    active          = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)


class ChunkSnapshot(Base):
    """
    Stores a copy of old chunks before repair, enabling rollback.
    Written by repair/reembedder.py before deleting old vectors.
    """
    __tablename__ = "autorag_chunk_snapshots"
    id          = Column(Integer, primary_key=True)
    event_id    = Column(Integer)              # links to the repair event
    vector_id   = Column(String(200))           # Pinecone vector ID
    text        = Column(Text)                  # original chunk text
    metadata_json = Column(Text)                # JSON of original metadata
    namespace   = Column(String(100))
    created_at  = Column(DateTime, default=datetime.utcnow)


class AdaptationLog(Base):
    """
    Full provenance record for every adaptation cycle.
    Records: what was observed → what was decided → what was changed → what resulted.
    Written by repair/orchestrator.py after each repair attempt.
    """
    __tablename__ = "autorag_adaptation_log"
    id                = Column(Integer, primary_key=True)
    event_id          = Column(Integer)
    observation       = Column(Text)          # JSON: which metrics failed, their values
    diagnosis         = Column(Text)          # JSON: root cause, question category
    strategy_selected = Column(String(50))
    config_before     = Column(Text)          # JSON: old PipelineConfig
    config_after      = Column(Text)          # JSON: new PipelineConfig
    metrics_before    = Column(Text)          # JSON: pre-change metric values
    metrics_after     = Column(Text)          # JSON: post-change metric values
    outcome           = Column(String(20))    # IMPROVED | DEGRADED | NO_CHANGE
    rolled_back       = Column(Boolean, default=False)
    cooldown_until    = Column(DateTime, nullable=True)
    created_at        = Column(DateTime, default=datetime.utcnow)
