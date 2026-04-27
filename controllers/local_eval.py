import json
import re
import string
import csv
import numpy as np
import requests
from datetime import datetime
from fastapi import UploadFile

from controllers.retrieval import answer_query
from services.llm_factory import get_embeddings
from config import settings
from services.query_logger import log_interaction

embedder = get_embeddings()

# --- Helper Functions ---
def get_embedding(text: str) -> np.ndarray:
    """Get embedding seamlessly from whatever provider is in .env"""
    vector = embedder.embed_query(text[:2000]) 
    return np.array(vector)

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

# --- NEW: Context Similarity Helper ---
def context_similarity(question: str, ground_truths: list, contexts: list) -> dict:
    """Evaluates context relevance using local HuggingFace embeddings."""
    if not contexts:
        return {
            "ctx_question_sim": float("nan"), "ctx_ground_truth_sim": float("nan"),
            "best_ctx_question_sim": float("nan"), "best_ctx_gt_sim": float("nan"),
        }
    try:
        q_emb = get_embedding(question)
        gt_embs = [get_embedding(gt) for gt in ground_truths]

        q_scores = []
        gt_scores = []

        for ctx in contexts:
            ctx_emb = get_embedding(ctx[:2000])
            
            # 1. Compare context against the question
            q_sim = float(np.dot(q_emb, ctx_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(ctx_emb)))
            q_scores.append(q_sim)
            
            # 2. Compare context against ALL ground truths (take the best match)
            current_ctx_gt_sims = [
                float(np.dot(gt_emb, ctx_emb) / (np.linalg.norm(gt_emb) * np.linalg.norm(ctx_emb)))
                for gt_emb in gt_embs
            ]
            gt_scores.append(max(current_ctx_gt_sims) if current_ctx_gt_sims else 0.0)

        return {
            "ctx_question_sim": float(np.mean(q_scores)) if q_scores else 0.0,
            "ctx_ground_truth_sim": float(np.mean(gt_scores)) if gt_scores else 0.0,
            "best_ctx_question_sim": max(q_scores) if q_scores else 0.0,
            "best_ctx_gt_sim": max(gt_scores) if gt_scores else 0.0,
        }
    except Exception as e:
        print(f"    WARNING: Context similarity failed: {e}")
        return {
            "ctx_question_sim": float("nan"), "ctx_ground_truth_sim": float("nan"),
            "best_ctx_question_sim": float("nan"), "best_ctx_gt_sim": float("nan"),
        }


# --- Core API Logic ---
async def run_local_evaluation(file: UploadFile, namespace: str, max_questions: int):
    contents = await file.read()
    raw_data = json.loads(contents)
    
    results = []
    questions_to_process = raw_data[:max_questions]
    total_q = len(questions_to_process)

    print(f"\n--- 🚀 STARTING LOCAL EVALUATION ---")
    print(f"📁 Namespace: '{namespace}' | ❓ Max Questions: {max_questions}")

    for i, item in enumerate(questions_to_process, 1):
        q = item["qun"]
        ground_truths = item["ans"]
        
        print(f"⏳ Processing [{i}/{total_q}]: {q[:50]}...")
        
        try:
            # 1. ADD skip_log=True to prevent saving NULL metrics
            rag_result = answer_query(q, namespace=namespace, skip_log=True)
            
            answer = rag_result["answer"]
            retrieved_contexts = rag_result.get("retrieved_contexts", [])
            num_contexts = len(retrieved_contexts)
            q_scores = rag_result.get("scores", [])  # <-- Extract the raw Pinecone scores
            
            rl_scores = [rouge_l(answer, gt) for gt in ground_truths]
            sem_scores = [semantic_similarity(answer, gt) for gt in ground_truths]
            
            best_rl = max(rl_scores) if rl_scores else 0.0
            
            valid_sem = [s for s in sem_scores if not np.isnan(s)]
            best_sem = max(valid_sem) if valid_sem else float('nan')
            
            # Calculate Context Similarities
            ctx_sims = context_similarity(q, ground_truths, retrieved_contexts)
            
            # 2. PACKAGE metrics for the database
            metrics_dict = {
                "rouge_l": best_rl,
                "semantic_similarity": best_sem,
                "ctx_question_sim": ctx_sims.get("ctx_question_sim", 0.0),
                "ctx_ground_truth_sim": ctx_sims.get("ctx_ground_truth_sim", 0.0),
                "best_ctx_question_sim": ctx_sims.get("best_ctx_question_sim", 0.0),
                "best_ctx_gt_sim": ctx_sims.get("best_ctx_gt_sim", 0.0)
            }

            # 3. LOG the complete interaction with the metrics
            log_interaction(
                namespace=namespace,
                query=q,
                answer=answer,
                contexts=retrieved_contexts,
                scores=q_scores,
                eval_metrics=metrics_dict
            )
            
            # Safely extract up to 3 ground truths for the CSV
            gt_1 = ground_truths[0] if len(ground_truths) > 0 else ""
            gt_2 = ground_truths[1] if len(ground_truths) > 1 else ""
            gt_3 = ground_truths[2] if len(ground_truths) > 2 else ""
            
            results.append({
                "question": q,
                "ground_truth_1": gt_1,
                "ground_truth_2": gt_2,
                "ground_truth_3": gt_3,
                "answer": answer,
                "num_contexts": num_contexts,
                "rouge_l": best_rl,
                "semantic_similarity": best_sem,
                **ctx_sims 
            })
            
            print(f"   ✅ Done! Best ROUGE-L: {best_rl:.4f} | Best Ctx-GT Sim: {ctx_sims['best_ctx_gt_sim']:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error on question {i}: {e}")
            continue

    if not results:
        print("❌ Evaluation failed. No questions processed.")
        return {"error": "No questions could be processed."}

    print("\n📊 Calculating averages and saving files...")

    if not results:
        print("❌ Evaluation failed. No questions processed.")
        return {"error": "No questions could be processed."}

    print("\n📊 Calculating averages and saving files...")

    # Safe average calculation helper
    def safe_avg(key):
        vals = [r.get(key, 0) for r in results if not np.isnan(r.get(key, 0))]
        return sum(vals) / len(vals) if vals else 0.0

    avg_rl = safe_avg("rouge_l")
    avg_ss = safe_avg("semantic_similarity")
    avg_ctx_q = safe_avg("ctx_question_sim")
    avg_ctx_gt = safe_avg("ctx_ground_truth_sim")
    avg_best_ctx_q = safe_avg("best_ctx_question_sim")
    avg_best_ctx_gt = safe_avg("best_ctx_gt_sim")

    # --- Generate CSV File (Updated Columns) ---
    csv_file = "evaluation_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question", 
            "ground_truth_1", "ground_truth_2", "ground_truth_3", 
            "answer", "num_contexts", "rouge_l", "semantic_similarity",
            "ctx_question_sim", "ctx_ground_truth_sim",        # <-- NEW
            "best_ctx_question_sim", "best_ctx_gt_sim"         # <-- NEW
        ])
        writer.writeheader()
        writer.writerows(results)

    # --- Generate Summary JSON File ---
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm": settings.llm_model_name,
            "llm_provider": getattr(settings, "llm_provider", "ollama"),                 
            "embedding_model": settings.embedding_model_name,
            "embedding_provider": getattr(settings, "embedding_provider", "ollama"),     
            "vector_db": f"Pinecone ({settings.pinecone_index_name})",
            "namespace_tested": namespace
        },
        "num_questions": len(results),
        "averages": {
            "rouge_l": round(float(avg_rl), 4),
            "semantic_similarity": round(float(avg_ss), 4),
            "ctx_question_sim": round(float(avg_ctx_q), 4),           # <-- NEW
            "ctx_ground_truth_sim": round(float(avg_ctx_gt), 4),      # <-- NEW
            "best_ctx_question_sim": round(float(avg_best_ctx_q), 4), # <-- NEW
            "best_ctx_gt_sim": round(float(avg_best_ctx_gt), 4)       # <-- NEW
        }
    }

    with open("evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Successfully saved {csv_file} and evaluation_summary.json")
    print("--- 🎉 EVALUATION COMPLETE ---\n")

    return summary