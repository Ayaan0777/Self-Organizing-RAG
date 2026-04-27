import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# Start with SQLite for prototyping; easy to swap to PostgreSQL later
SQLALCHEMY_DATABASE_URL = "sqlite:///./query_logs.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    namespace = Column(String, index=True)
    user_query = Column(Text)
    llm_answer = Column(Text)
    retrieved_contexts = Column(Text) 
    similarity_scores = Column(Text)  

    # --- NEW EVALUATION METRICS ---
    rouge_l = Column(Float, nullable=True)
    semantic_similarity = Column(Float, nullable=True)
    ctx_question_sim = Column(Float, nullable=True)
    ctx_ground_truth_sim = Column(Float, nullable=True)
    best_ctx_question_sim = Column(Float, nullable=True)
    best_ctx_gt_sim = Column(Float, nullable=True)

# Create the table
Base.metadata.create_all(bind=engine)

def log_interaction(
    namespace: str, 
    query: str, 
    answer: str, 
    contexts: list, 
    scores: list,
    eval_metrics: dict = None  # <-- Added an optional dictionary for metrics
):
    """Saves the RAG interaction to the database."""
    db = SessionLocal()
    try:
        log_entry = QueryLog(
            namespace=namespace,
            user_query=query,
            llm_answer=answer,
            retrieved_contexts=json.dumps(contexts),
            similarity_scores=json.dumps(scores)
        )
        
        # If metrics are provided (e.g., during an evaluation run), map them to the DB columns
        if eval_metrics:
            log_entry.rouge_l = eval_metrics.get("rouge_l")
            log_entry.semantic_similarity = eval_metrics.get("semantic_similarity")
            log_entry.ctx_question_sim = eval_metrics.get("ctx_question_sim")
            log_entry.ctx_ground_truth_sim = eval_metrics.get("ctx_ground_truth_sim")
            log_entry.best_ctx_question_sim = eval_metrics.get("best_ctx_question_sim")
            log_entry.best_ctx_gt_sim = eval_metrics.get("best_ctx_gt_sim")

        db.add(log_entry)
        db.commit()
    except Exception as e:
        print(f"❌ Failed to log query: {e}")
    finally:
        db.close()

def get_recent_logs(limit: int = 50):
    """Fetches recent logs for the diagnostic dashboard."""
    db = SessionLocal()
    try:
        logs = db.query(QueryLog).order_by(QueryLog.timestamp.desc()).limit(limit).all()
        return [
            {
                "id": log.id,
                "timestamp": log.timestamp.isoformat(),
                "namespace": log.namespace,
                "query": log.user_query,
                "answer": log.llm_answer,
                "contexts": json.loads(log.retrieved_contexts),
                "scores": json.loads(log.similarity_scores),
                # Include metrics in the fetch response
                "metrics": {
                    "rouge_l": log.rouge_l,
                    "semantic_similarity": log.semantic_similarity,
                    "ctx_question_sim": log.ctx_question_sim,
                    "ctx_ground_truth_sim": log.ctx_ground_truth_sim,
                    "best_ctx_question_sim": log.best_ctx_question_sim,
                    "best_ctx_gt_sim": log.best_ctx_gt_sim,
                }
            }
            for log in logs
        ]
    finally:
        db.close()