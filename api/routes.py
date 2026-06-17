from fastapi import APIRouter, UploadFile, File, Query, Form,HTTPException
from pydantic import BaseModel
from controllers import ingestion, retrieval, evaluation
from controllers.evaluation import process_local_evaluation
from repair.cascade import run_repair_cascade
router = APIRouter()


class QueryReq(BaseModel):
    query: str
    namespace: str = None


class EvalReq(BaseModel):
    question: str
    ground_truth: str


# ── Core RAG endpoints ────────────────────────────────────────
@router.post("/ingest")
async def ingest_endpoint(
    file: UploadFile = File(...),
    namespace: str = Query(None, description="Pinecone namespace"),
):
    return ingestion.process_and_store_file(file, namespace)


@router.post("/query")
async def query_endpoint(req: QueryReq):
    return retrieval.answer_query(req.query, req.namespace)


@router.post("/evaluate")
async def eval_endpoint(req: EvalReq):
    return evaluation.calculate_metrics(req.question, req.ground_truth)


# ── Add Chunks endpoint ───────────────────────────────────────
class AddChunksReq(BaseModel):
    text: str
    source: str = "manual"
    ingest: bool = False             # if True, also upload to Pinecone
    namespace: str = None


@router.post("/add-chunks")
async def add_chunks_endpoint(req: AddChunksReq):
    """Chunks raw text using the same recursive splitter as ingestion
    (variable size 500–1250 chars). Optionally ingests to Pinecone."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from controllers.ingestion import (
        CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE,
        SEPARATORS, _enforce_min_chunk_size,
    )

    # Exact same parameters as ingestion pipeline
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    chunks = splitter.create_documents(
        [req.text],
        metadatas=[{"source": req.source, "strategy": "recursive"}],
    )

    # Enforce minimum chunk size — same as ingestion
    chunks = _enforce_min_chunk_size(chunks, min_chars=MIN_CHUNK_SIZE)

    sizes = [len(c.page_content) for c in chunks]
    result = {
        "strategy": "recursive",
        "num_chunks": len(chunks),
        "size_range": f"{min(sizes)}-{max(sizes)}" if sizes else "0",
        "chunks": [
            {"content": c.page_content[:300], "chars": len(c.page_content)}
            for c in chunks
        ],
    }

    if req.ingest:
        from services.llm_factory import get_vector_store
        vs = get_vector_store(req.namespace)
        vs.add_documents(chunks)
        result["ingested"] = True
        result["namespace"] = req.namespace or "default"

    return result


# ── Auto-RAG monitoring endpoints ─────────────────────────────
import json as _json
from db.session import get_session
from db.models import QueryLog, LowRecallEvent, RepairReport, AdaptationLog, StrategyCounter, RuntimeFlag


@router.get("/logs")
def get_logs(limit: int = 50):
    """Returns recent query log entries with scores and flagging status."""
    s    = get_session()
    rows = s.query(QueryLog).order_by(QueryLog.timestamp.desc()).limit(limit).all()
    s.close()
    return [
        {
            "id"        : r.id,
            "query"     : r.query,
            "scores"    : _json.loads(r.top_k_scores or "[]"),
            "flagged"   : r.flagged,
            "latency_ms": r.latency_ms,
            "ts"        : str(r.timestamp),
        }
        for r in rows
    ]


@router.get("/events")
async def get_events(unresolved_only: bool = True):
    """
    Returns a list of LowRecallEvents.
    By default, hides events that have already been successfully repaired.
    """
    from db.session import get_session
    from db.models import LowRecallEvent
    
    session = get_session()
    try:
        query = session.query(LowRecallEvent)
        
        # Filter out the repaired events so we only see the active "hit list"
        if unresolved_only:
            query = query.filter(LowRecallEvent.resolved == False)
            
        events = query.order_by(LowRecallEvent.timestamp.desc()).limit(100).all()
        
        return [
            {
                "id": e.id,
                "query_log_id": e.query_log_id,
                "severity": e.severity,
                "detectors": e.triggered_detectors,
                "resolved": e.resolved,
                "timestamp": e.timestamp
            }
            for e in events
        ]
    finally:
        session.close()


# ── Month 4: Auto Indexer endpoints ───────────────────────────

@router.get("/index/health")
def index_health(namespace: str = None):
    """Runs consistency checks on the Pinecone index."""
    from auto_indexer.engine import AutoIndexer
    indexer = AutoIndexer(namespace)
    return indexer.check_consistency()


@router.get("/index/staleness")
def index_staleness(namespace: str = None, sample_size: int = 30):
    """Detects stale embeddings by comparing stored vs fresh embeddings."""
    from auto_indexer.engine import AutoIndexer
    indexer = AutoIndexer(namespace)
    return indexer.detect_stale_chunks(sample_size=sample_size)


@router.post("/index/refresh")
def index_refresh(namespace: str = None, sample_size: int = 50, auto_fix: bool = True):
    """Runs the full auto-indexer pipeline: detect stale → re-embed → verify."""
    from auto_indexer.engine import AutoIndexer
    indexer = AutoIndexer(namespace)
    return indexer.run_full_refresh(sample_size=sample_size, auto_fix=auto_fix)

@router.post("/evaluate-local")
async def evaluate_local_endpoint(
    file: UploadFile = File(...),
    namespace: str = Form("mxbai-embed-large"),
    max_questions: int = Form(30),
    start_index: int = Form(0)
):
    """
    Upload a JSON dataset to run a fully local Ollama evaluation.
    Requires fields: 'qun' (question string) and 'ans' (list of ground truth strings).
    """
    return await process_local_evaluation(
        file=file, 
        namespace=namespace, 
        max_questions=max_questions, 
        start_index=start_index
    )
@router.post("/repair/{event_id}")
async def trigger_repair_loop(event_id: int):
    """
    Triggers the ordered repair cascade for a specific LowRecallEvent.
    Tries S1 (dynamic K) → S2 (chunk size) → S3 (combined) → S4 (alt LLM).
    First strategy that resolves the issue wins.
    """
    result = run_repair_cascade(event_id)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    return result


@router.get("/repair-report/{event_id}")
def get_repair_report(event_id: int):
    """Returns the latest repair report + diagnosis for a LowRecallEvent."""
    s = get_session()
    try:
        report = (
            s.query(RepairReport)
            .filter(RepairReport.event_id == event_id)
            .order_by(RepairReport.timestamp.desc())
            .first()
        )
        if not report:
            raise HTTPException(status_code=404, detail="Repair report not found")

        # Fetch the adaptation log for diagnosis/reasoning
        adaptation = (
            s.query(AdaptationLog)
            .filter(AdaptationLog.event_id == event_id)
            .order_by(AdaptationLog.created_at.desc())
            .first()
        )

        diagnosis_info = {}
        if adaptation:
            try:
                diag = _json.loads(adaptation.diagnosis or "{}")
                diagnosis_info = {
                    "root_cause": diag.get("root_cause", "unknown"),
                    "question_category": diag.get("question_category", "unknown"),
                    "severity_score": diag.get("severity_score", 0),
                    "reasoning": diag.get("reasoning", ""),
                }
            except Exception:
                pass

        # If resolved but resolved_answer is missing, generate it on-the-fly
        # and backfill the DB so it's only done once
        resolved_answer = report.resolved_answer
        if report.resolved and not resolved_answer:
            try:
                event = s.query(LowRecallEvent).filter(
                    LowRecallEvent.id == event_id
                ).first()
                if event:
                    log = s.query(QueryLog).filter(
                        QueryLog.id == event.query_log_id
                    ).first()
                    if log:
                        from controllers.retrieval import generate_answer_only
                        rag_result = generate_answer_only(log.query)
                        resolved_answer = rag_result.get("answer", "")
                        # Backfill the DB record
                        if resolved_answer:
                            report.resolved_answer = resolved_answer
                            s.commit()
            except Exception as e:
                import logging
                logging.warning(f"[repair-report] On-the-fly answer generation failed: {e}")

        return {
            "event_id": report.event_id,
            "strategy_used": report.strategy_used,
            "score_before": report.score_before,
            "score_after": report.score_after,
            "resolved": report.resolved,
            "original_answer": report.original_answer,
            "resolved_answer": resolved_answer,
            # Enhanced metrics
            "precision_before": report.precision_before,
            "precision_after": report.precision_after,
            "recall_before": report.recall_before,
            "recall_after": report.recall_after,
            "accuracy_before": report.accuracy_before,
            "accuracy_after": report.accuracy_after,
            # Dynamic K + chunk comparison
            "dynamic_k": report.dynamic_k,
            "chunks_before_text": _json.loads(report.chunks_before_text or "[]"),
            "chunks_after_text": _json.loads(report.chunks_after_text or "[]"),
            # Diagnosis
            **diagnosis_info,
        }
    finally:
        s.close()


@router.get("/eval-history")
def get_eval_history(limit: int = 100):
    """Returns recent evaluation snapshots for the dashboard."""
    from db.models import EvalSnapshot
    s = get_session()
    rows = s.query(EvalSnapshot).order_by(EvalSnapshot.timestamp.desc()).limit(limit).all()
    s.close()
    return [
        {
            "id": r.id,
            "namespace": r.namespace,
            "llm": r.llm,
            "embeddings": r.embeddings,
            "rouge_l": r.rouge_l,
            "sem_sim": r.sem_sim,
            "ctx_q_sim": r.ctx_q_sim,
            "ctx_gt_sim": r.ctx_gt_sim,
            "retrieval_precision": r.retrieval_precision,
            "context_sufficiency": r.context_sufficiency,
            "hallucination_rate": r.hallucination_rate,
            "timestamp": str(r.timestamp),
        }
        for r in rows
    ]


# ── Stage 2-4: Pipeline Config & Adaptation Log endpoints ─────

@router.get("/pipeline-config")
def get_pipeline_config(namespace: str = None):
    """Returns the current active pipeline configuration."""
    from detector.decision_engine import get_active_config
    config = get_active_config(namespace)
    return config


@router.post("/pipeline-config")
def set_pipeline_config(
    chunk_size: int = Query(250, description="New chunk size"),
    chunk_overlap: int = Query(80, description="New chunk overlap"),
    chunk_strategy: str = Query("semantic", description="Chunking strategy"),
    namespace: str = Query(None, description="Pinecone namespace"),
):
    """Manually override the pipeline configuration (for testing)."""
    from detector.decision_engine import save_new_config
    from config import settings
    ns = namespace or settings.pinecone_namespace
    config_id = save_new_config(ns, chunk_size, chunk_overlap, chunk_strategy)
    return {
        "message": "Pipeline config updated",
        "config_id": config_id,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_strategy": chunk_strategy,
        "namespace": ns,
    }


@router.get("/adaptation-log")
def get_adaptation_log(limit: int = 50):
    """
    Returns the adaptation provenance trail — full audit of every
    self-healing cycle: what was observed, decided, changed, and resulted.
    """
    from db.models import AdaptationLog
    s = get_session()
    rows = s.query(AdaptationLog).order_by(AdaptationLog.created_at.desc()).limit(limit).all()
    s.close()
    return [
        {
            "id": r.id,
            "event_id": r.event_id,
            "observation": r.observation,
            "diagnosis": r.diagnosis,
            "strategy_selected": r.strategy_selected,
            "config_before": r.config_before,
            "config_after": r.config_after,
            "metrics_before": r.metrics_before,
            "metrics_after": r.metrics_after,
            "outcome": r.outcome,
            "rolled_back": r.rolled_back,
            "created_at": str(r.created_at),
        }
        for r in rows
    ]


# ── Cascade: Strategy Counters & Runtime Flags ────────────────

@router.get("/strategy-counters")
def get_strategy_counters():
    """Returns all strategy success counters for the repair cascade."""
    s = get_session()
    try:
        counters = s.query(StrategyCounter).all()
        return [
            {
                "strategy": c.strategy,
                "success_count": c.success_count,
                "last_incremented_at": str(c.last_incremented_at) if c.last_incremented_at else None,
            }
            for c in counters
        ]
    finally:
        s.close()


@router.get("/runtime-flags")
def get_runtime_flags():
    """Returns all runtime flags (e.g., dynamic_k_promoted)."""
    s = get_session()
    try:
        flags = s.query(RuntimeFlag).all()
        return [
            {
                "name": f.name,
                "value": f.value,
                "set_at": str(f.set_at) if f.set_at else None,
            }
            for f in flags
        ]
    finally:
        s.close()

