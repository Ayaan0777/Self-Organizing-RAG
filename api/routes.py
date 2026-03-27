from fastapi import APIRouter, UploadFile, File,Form
from pydantic import BaseModel
from controllers import ingestion, retrieval, evaluation

router = APIRouter()

class QueryReq(BaseModel):
    query: str
    namespace:str = "default"  # <-- Add namespace to query request

class EvalReq(BaseModel):
    question: str
    ground_truth: str

# 1. This endpoint now expects a file upload
@router.post("/ingest")
async def ingest_endpoint(
    file: UploadFile = File(...),
    namespace: str = Form("default")):  # <-- Accept namespace as form data           
    return ingestion.process_and_store_file(file, namespace=namespace)

@router.post("/query")
async def query_endpoint(req: QueryReq):
    return retrieval.answer_query(req.query,req.namespace)

@router.post("/evaluate")
async def eval_endpoint(req: EvalReq):
    return evaluation.calculate_metrics(req.question, req.ground_truth)