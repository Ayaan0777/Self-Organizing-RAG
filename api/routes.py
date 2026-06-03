from fastapi import APIRouter, UploadFile, File, Query
from pydantic import BaseModel
from controllers import ingestion, retrieval, evaluation

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


# ── Auto-Chunker endpoint ─────────────────────────────────────
class AutoChunkReq(BaseModel):
    text: str
    source: str = "manual"
    strategy: str = "semantic"       # "semantic" or "clustering"
    ingest: bool = False             # if True, also upload to Pinecone
    namespace: str = None


@router.post("/auto-chunk")
async def auto_chunk_endpoint(req: AutoChunkReq):
    """Runs the auto-chunker pipeline. Optionally ingests results to Pinecone."""
    from auto_chunker import auto_chunk
    chunks = auto_chunk(req.text, req.source, strategy=req.strategy)

    result = {
        "strategy": req.strategy,
        "num_chunks": len(chunks),
        "chunks": [{"content": c.page_content[:200], "chars": len(c.page_content)} for c in chunks],
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
def get_events():
    """Returns all low-recall events ordered by most recent."""
    s    = get_session()
    rows = s.query(LowRecallEvent).order_by(LowRecallEvent.timestamp.desc()).all()
    s.close()
    return [
        {
            "id"       : r.id,
            "severity" : r.severity,
            "detectors": _json.loads(r.triggered_detectors or "[]"),
            "resolved" : r.resolved,
            "ts"       : str(r.timestamp),
        }
        for r in rows
    ]




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