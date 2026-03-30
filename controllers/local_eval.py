import json
import re
import string
import csv
import numpy as np
import requests
from datetime import datetime
from fastapi import UploadFile

from controllers.retrieval import answer_query
from config import settings

# --- Helper Functions ---
def get_embedding(text: str) -> np.ndarray:
    url = f"{settings.ollama_base_url}/api/embed"
    resp = requests.post(url, json={"model": settings.embedding_model_name, "input": text[:2000]}, timeout=60)
    resp.raise_for_status()
    return np.array(resp.json()["embeddings"][0])

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

# --- Core API Logic ---
# --- Core API Logic ---
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
            rag_result = answer_query(q, namespace=namespace)
            answer = rag_result["answer"]
            num_contexts = len(rag_result.get("retrieved_contexts", []))
            
            rl_scores = [rouge_l(answer, gt) for gt in ground_truths]
            sem_scores = [semantic_similarity(answer, gt) for gt in ground_truths]
            
            best_rl = max(rl_scores) if rl_scores else 0.0
            
            valid_sem = [s for s in sem_scores if not np.isnan(s)]
            best_sem = max(valid_sem) if valid_sem else float('nan')
            
            # --- NEW: Safely extract up to 3 ground truths for the CSV ---
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
                "semantic_similarity": best_sem
            })
            
            print(f"   ✅ Done! Best ROUGE-L: {best_rl:.4f} | Best SemSim: {best_sem:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error on question {i}: {e}")
            continue

    if not results:
        print("❌ Evaluation failed. No questions processed.")
        return {"error": "No questions could be processed."}

    print("\n📊 Calculating averages and saving files...")

    avg_rl = sum(r["rouge_l"] for r in results) / len(results)
    avg_ss = sum(r["semantic_similarity"] for r in results) / len(results)

    # --- Generate CSV File (Updated Columns) ---
    csv_file = "evaluation_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question", 
            "ground_truth_1", "ground_truth_2", "ground_truth_3", 
            "answer", "num_contexts", "rouge_l", "semantic_similarity"
        ])
        writer.writeheader()
        writer.writerows(results)

    # --- Generate Summary JSON File ---
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm": settings.llm_model_name,
            "embedding_model": settings.embedding_model_name,
            "vector_db": f"Pinecone ({settings.pinecone_index_name})",
            "provider": "Ollama (fully local)",
            "namespace_tested": namespace
        },
        "num_questions": len(results),
        "averages": {
            "rouge_l": round(float(avg_rl), 4),
            "semantic_similarity": round(float(avg_ss), 4)
        }
    }

    with open("evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Successfully saved {csv_file} and evaluation_summary.json")
    print("--- 🎉 EVALUATION COMPLETE ---\n")

    return summary