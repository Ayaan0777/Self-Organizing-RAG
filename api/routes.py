from fastapi import APIRouter, UploadFile, File, Query, Form,HTTPException
from pydantic import BaseModel
from controllers import ingestion, retrieval, evaluation
from controllers.evaluation import process_local_evaluation
from repair.orchestrator import handle_event
router = APIRouter()


class QueryReq(BaseModel):
    query: str
    namespace: str = None
    k: int = 5                        # Phase 2: dynamic top-K
    rerank: bool = True                # Phase 2: LLM reranking
    metadata_filter: dict = None       # Phase 2: metadata filtering


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
    return retrieval.answer_query(
        query=req.query,
        namespace=req.namespace,
        k=req.k,
        rerank=req.rerank,
        metadata_filter=req.metadata_filter,
    )


@router.post("/evaluate")
async def eval_endpoint(req: EvalReq):
    return evaluation.calculate_metrics(req.question, req.ground_truth)


# ── Add Chunks endpoint ───────────────────────────────────────
class AutoChunkReq(BaseModel):
    text: str
    source: str = "manual"
    ingest: bool = False             # if True, also upload to Pinecone
    namespace: str = None


@router.post("/auto-chunk")
async def auto_chunk_endpoint(req: AutoChunkReq):
    """Chunks raw text using recursive splitting. Optionally ingests to Pinecone."""
    from auto_chunker import auto_chunk
    chunks = auto_chunk(req.text, req.source)

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
from db.models import QueryLog, LowRecallEvent, RepairReport


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
    Triggers the self-healing repair loop for a specific LowRecallEvent.
    Strategy is auto-selected by the DECIDE stage based on:
      - Which detectors triggered (context_insufficient, hallucination_detected)
      - Query complexity (complex → bigger chunks, simple → smaller chunks)
    Includes rollback: if repair doesn't improve, old chunks are restored.
    """
    result = handle_event(event_id)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.get("/repair-reports")
def get_repair_reports(limit: int = 100):
    """Returns repair history for the dashboard — shows all repair attempts."""
    s = get_session()
    rows = s.query(RepairReport).order_by(RepairReport.timestamp.desc()).limit(limit).all()
    s.close()
    return [
        {
            "id":             r.id,
            "event_id":       r.event_id,
            "strategy":       r.strategy_used,
            "chunk_size":     r.chunk_size_used,
            "repair_reason":  r.repair_reason,
            "score_before":   r.score_before,
            "score_after":    r.score_after,
            "chunks_before":  r.chunks_before,
            "chunks_after":   r.chunks_after,
            "resolved":       r.resolved,
            "rolled_back":    r.rolled_back,
            "duration_ms":    r.duration_ms,
            "timestamp":      str(r.timestamp),
        }
        for r in rows
    ]


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
            "timestamp": str(r.timestamp),
        }
        for r in rows
    ]


# ── Phase 4: Feedback Analysis endpoint ───────────────────────
@router.get("/feedback/analysis")
def feedback_analysis():
    """Returns a comprehensive analysis of detection + repair effectiveness."""
    from organiser.feedback import analyse_system_health
    return analyse_system_health()
