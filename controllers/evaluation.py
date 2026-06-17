import json
import re
import string
import csv
import os
import numpy as np
from datetime import datetime
from fastapi import UploadFile

from controllers.retrieval import answer_query
from config import settings
from services.llm_factory import get_embeddings
from db.session import get_session
from db.models import EvalSnapshot
from logger.query_logger import update_log_eval_metrics, update_log_new_metrics
from controllers.metrics import (
    retrieval_precision_at_k,
    context_sufficiency as compute_context_sufficiency,
    hallucination_rate as compute_hallucination_rate,
    classify_question,
)

# ──────────────────────────────────────────────
# OLLAMA EMBEDDING HELPER
# ──────────────────────────────────────────────
def get_embedding(text: str) -> np.ndarray:
    """Get embedding from local Ollama server via LangChain."""
    emb_model = get_embeddings()
    vector = emb_model.embed_query(text[:500])
    return np.array(vector)

# ══════════════════════════════════════════════
#  METRIC FUNCTIONS
# ══════════════════════════════════════════════
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)

def rouge_l(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize(prediction).split()
    gt_tokens = normalize(ground_truth).split()

    if not pred_tokens and not gt_tokens: return 1.0
    if not pred_tokens or not gt_tokens: return 0.0

    m, n = len(pred_tokens), len(gt_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gt_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    if lcs_len == 0: return 0.0
    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * (precision * recall) / (precision + recall)

def semantic_similarity(text1: str, text2: str) -> float:
    try:
        v1, v2 = get_embedding(text1), get_embedding(text2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    except Exception:
        return float("nan")

def context_similarity(question: str, ground_truths: list, contexts: list) -> dict:
    try:
        q_emb = get_embedding(question)
        gt_embs = [get_embedding(gt) for gt in ground_truths]

        q_scores, gt_scores = [], []
        for ctx in contexts:
            ctx_emb = get_embedding(ctx)
            q_sim = float(np.dot(q_emb, ctx_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(ctx_emb)))
            
            gt_sims = [float(np.dot(gt_emb, ctx_emb) / (np.linalg.norm(gt_emb) * np.linalg.norm(ctx_emb))) for gt_emb in gt_embs]
            q_scores.append(q_sim)
            gt_scores.append(max(gt_sims) if gt_sims else 0.0)

        return {
            "ctx_question_sim": float(np.mean(q_scores)) if q_scores else 0.0,
            "ctx_ground_truth_sim": float(np.mean(gt_scores)) if gt_scores else 0.0,
            "best_ctx_question_sim": max(q_scores) if q_scores else 0.0,
            "best_ctx_gt_sim": max(gt_scores) if gt_scores else 0.0,
        }
    except Exception:
        return {
            "ctx_question_sim": float("nan"), "ctx_ground_truth_sim": float("nan"),
            "best_ctx_question_sim": float("nan"), "best_ctx_gt_sim": float("nan"),
        }

# ══════════════════════════════════════════════
#  CORE API FUNCTION (CRITICAL FIX HERE)
# ══════════════════════════════════════════════
async def process_local_evaluation(file: UploadFile, namespace: str, max_questions: int, start_index: int = 0):
    # 1. Read and parse the uploaded file
    contents = await file.read()
    raw_data = json.loads(contents)
    
    test_data = []
    # Clean slice using python array slicing: [start : start + length]
    for item in raw_data[start_index : start_index + max_questions]:
        test_data.append({
            "question": item.get("qun", ""),
            "ground_truths": item.get("ans", [])
        })

    results = []
    print(f"\n🚀 Starting Evaluation for namespace '{namespace}' on {len(test_data)} questions starting at index {start_index}...")

    # 2. Run the RAG pipeline on each question
    for i, item in enumerate(test_data, 1):
        q = item["question"]
        gts = item["ground_truths"]
        print(f"  [{i}/{len(test_data)}] Processing: {q[:50]}...")

        try:
            # Execute query on the baseline RAG pipeline
            rag_result = answer_query(q, namespace=namespace)
            answer = rag_result["answer"]
            retrieved_contexts = rag_result["retrieved_contexts"]
            log_id = rag_result.get("log_id", -1)

            rl = max((rouge_l(answer, gt) for gt in gts), default=0.0)
            ss = max((semantic_similarity(answer, gt) for gt in gts), default=0.0)
            ctx_sims = context_similarity(q, gts, retrieved_contexts)

            # ── Stage 2: Compute new metrics ──
            ret_precision = retrieval_precision_at_k(retrieved_contexts, gts)
            ctx_suff = compute_context_sufficiency(retrieved_contexts, gts)
            hall_rate = compute_hallucination_rate(answer, retrieved_contexts)
            q_category = classify_question(q)
            print(f"      [metrics] precision={ret_precision:.2f} sufficiency={ctx_suff} "
                  f"hallucination={hall_rate:.2f} category={q_category}")

            try:
                update_log_eval_metrics(
                    log_id=log_id,
                    answer_sem_sim=ss,
                    ctx_q_sim=ctx_sims["ctx_question_sim"],
                )
                # Persist the new Stage 2 metrics
                update_log_new_metrics(
                    log_id=log_id,
                    retrieval_precision=ret_precision,
                    context_sufficiency=ctx_suff,
                    hallucination_rate=hall_rate,
                    question_category=q_category,
                )
            except Exception as ue:
                print(f"      [eval] could not update log metrics: {ue}")

            results.append({
                "question": q,
                "ground_truth": json.dumps(gts, ensure_ascii=False),
                "answer": answer,
                "num_contexts": len(retrieved_contexts),
                "rouge_l": rl,
                "semantic_similarity": ss,
                "retrieval_precision": ret_precision,
                "context_sufficiency": ctx_suff,
                "hallucination_rate": hall_rate,
                "question_category": q_category,
                **ctx_sims,
            })
        except Exception as e:
            print(f"    ❌ Error: {e}")
            continue

    if not results:
        return {"error": "Evaluation failed. No questions processed."}

    # 3. Aggregate results
    def safe_mean(values):
        clean = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
        return sum(clean) / len(clean) if clean else 0.0

    avg_rl = safe_mean([r["rouge_l"] for r in results])
    avg_ss = safe_mean([r["semantic_similarity"] for r in results])
    avg_ctx_q = safe_mean([r["ctx_question_sim"] for r in results])
    avg_ctx_gt = safe_mean([r["ctx_ground_truth_sim"] for r in results])
    # ── Stage 2: Aggregate new metrics ──
    avg_ret_prec = safe_mean([r["retrieval_precision"] for r in results])
    avg_ctx_suff = sum(1 for r in results if r["context_sufficiency"]) / len(results)
    avg_hall_rate = safe_mean([r["hallucination_rate"] for r in results])

    summary = {
        "timestamp": datetime.now().isoformat(),
        "namespace": namespace,
        "config": {
            "llm": settings.llm_model_name,
            "embedding_model": settings.embedding_model_name,
        },
        "num_questions": len(results),
        "averages": {
            "rouge_l": round(float(avg_rl), 4),
            "semantic_similarity": round(float(avg_ss), 4),
            "ctx_question_sim": round(float(avg_ctx_q), 4),
            "ctx_ground_truth_sim": round(float(avg_ctx_gt), 4),
            "retrieval_precision": round(float(avg_ret_prec), 4),
            "context_sufficiency": round(float(avg_ctx_suff), 4),
            "hallucination_rate": round(float(avg_hall_rate), 4),
        }
    }

    # 4. Save CSV to results directory
    os.makedirs("results", exist_ok=True)
    csv_file = f"results/evaluation_results_{namespace}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # 5. Persist to Dashboard DB
    try:
        session = get_session()
        snap = EvalSnapshot(
            namespace=namespace,
            llm=settings.llm_model_name,
            embeddings=settings.embedding_model_name,
            rouge_l=round(float(avg_rl), 4),
            sem_sim=round(float(avg_ss), 4),
            ctx_q_sim=round(float(avg_ctx_q), 4),
            ctx_gt_sim=round(float(avg_ctx_gt), 4),
            retrieval_precision=round(float(avg_ret_prec), 4),
            context_sufficiency=round(float(avg_ctx_suff), 4),
            hallucination_rate=round(float(avg_hall_rate), 4),
        )
        session.add(snap)
        session.commit()
        session.close()
    except Exception as e:
        print(f"[eval] could not save to DB: {e}")

    return {"message": "Evaluation complete", "summary": summary}