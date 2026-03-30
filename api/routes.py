from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from controllers import ingestion, retrieval, evaluation

router = APIRouter()

class QueryReq(BaseModel):
    query: str
    namespace:str = "default"

class EvalReq(BaseModel):
    question: str
    ground_truth: str

class ClearReq(BaseModel):
    confirm: bool = False

# 1. This endpoint now expects a file upload and an optional strategy
@router.post("/ingest")
async def ingest_endpoint(
    file: UploadFile = File(...), 
    strategy: str = Form("semantic"),
    namespace: str = Form("default")):
    return ingestion.process_and_store_file(file, strategy=strategy, namespace=namespace)

@router.post("/query")
async def query_endpoint(req: QueryReq):
    return retrieval.answer_query(req.query, req.namespace)

@router.post("/evaluate")
async def eval_endpoint(req: EvalReq):
    return evaluation.calculate_metrics(req.question, req.ground_truth)

@router.post("/clear")
async def clear_endpoint(req: ClearReq):
    """Clear all vectors from Pinecone. Requires confirm=true."""
    if not req.confirm:
        return {"error": "Must set 'confirm': true to clear the database"}
    return ingestion.clear_vector_store()