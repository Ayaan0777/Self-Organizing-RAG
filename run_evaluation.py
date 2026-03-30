"""
Lightweight RAG evaluation script (fully local - Ollama only).
Metrics: ROUGE-L, Semantic Similarity (Ollama embeddings).

Run:  python run_evaluation.py

Make sure:
  1. Ollama is running (ollama serve)
  2. Documents are already ingested via /ingest endpoint
  3. .env is properly configured
"""
import json
import re
import string
import csv
import numpy as np
import requests
from datetime import datetime

from controllers.retrieval import answer_query
from config import settings

# ──────────────────────────────────────────────
# MODEL → INDEX DIMENSION MAP
# Maps each embedding model to its Pinecone index.
# ──────────────────────────────────────────────
MODEL_INDEX_MAP = {
    "all-minilm":                  "rag-index",       # 384-dim
    "nomic-embed-text":            "rag-index-768",   # 768-dim
    "snowflake-arctic-embed":      "rag-index-1024",  # 1024-dim
    "snowflake-arctic-embed:335m": "rag-index-1024",
}

ACTIVE_MODEL = settings.embedding_model_name
ACTIVE_INDEX = MODEL_INDEX_MAP.get(ACTIVE_MODEL, settings.pinecone_index_name)

# ──────────────────────────────────────────────
# OLLAMA EMBEDDING HELPER
# ──────────────────────────────────────────────
OLLAMA_EMBED_URL = f"{settings.ollama_base_url}/api/embed"
OLLAMA_EMBED_MODEL = settings.embedding_model_name  # nomic-embed-text


def get_embedding(text: str) -> np.ndarray:
    """Get embedding from local Ollama server."""
    resp = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": OLLAMA_EMBED_MODEL, "input": text[:2000]},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns {"embeddings": [[...]]}
    return np.array(data["embeddings"][0])


# ──────────────────────────────────────────────
# LOAD DATA FROM JSON FILE
# ──────────────────────────────────────────────
QA_FILE = r"C:\Users\hegde\Downloads\d3.json"
MAX_QUESTIONS = 75

with open(QA_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

test_data = []
for item in raw_data[:MAX_QUESTIONS]:
    test_data.append({
        "question": item["qun"],
        "ground_truths": item["ans"]
    })

print(f"Loaded {len(test_data)} questions from {QA_FILE}")


# ══════════════════════════════════════════════
#  METRIC FUNCTIONS
# ══════════════════════════════════════════════

def normalize(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def rouge_l(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F1 score based on longest common subsequence."""
    pred_tokens = normalize(prediction).split()
    gt_tokens = normalize(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    # LCS via dynamic programming
    m, n = len(pred_tokens), len(gt_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gt_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * (precision * recall) / (precision + recall)


def semantic_similarity(text1: str, text2: str) -> float:
    """Cosine similarity using local Ollama embeddings."""
    try:
        v1 = get_embedding(text1)
        v2 = get_embedding(text2)
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(cos_sim)
    except Exception as e:
        print(f"    WARNING: Semantic similarity failed: {e}")
        return float("nan")


def context_similarity(question: str, ground_truths: list, contexts: list) -> dict:
    """
    Evaluate retrieval quality by measuring how semantically similar
    the retrieved contexts are to the question and ground truths.
    Returns avg similarity scores.
    """
    try:
        q_emb = get_embedding(question)
        gt_embs = [get_embedding(gt) for gt in ground_truths]

        q_scores = []
        gt_scores = []

        for ctx in contexts:
            ctx_emb = get_embedding(ctx[:2000])
            q_sim = float(np.dot(q_emb, ctx_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(ctx_emb)))
            
            gt_sims = [float(np.dot(gt_emb, ctx_emb) / (np.linalg.norm(gt_emb) * np.linalg.norm(ctx_emb))) for gt_emb in gt_embs]
            best_gt_sim = max(gt_sims) if gt_sims else 0.0
            
            q_scores.append(q_sim)
            gt_scores.append(best_gt_sim)

        return {
            "ctx_question_sim": float(np.mean(q_scores)) if q_scores else 0.0,
            "ctx_ground_truth_sim": float(np.mean(gt_scores)) if gt_scores else 0.0,
            "best_ctx_question_sim": max(q_scores) if q_scores else 0.0,
            "best_ctx_gt_sim": max(gt_scores) if gt_scores else 0.0,
        }
    except Exception as e:
        print(f"    WARNING: Context similarity failed: {e}")
        return {
            "ctx_question_sim": float("nan"),
            "ctx_ground_truth_sim": float("nan"),
            "best_ctx_question_sim": float("nan"),
            "best_ctx_gt_sim": float("nan"),
        }


# ══════════════════════════════════════════════
#  VERIFY OLLAMA IS RUNNING
# ══════════════════════════════════════════════
print("\nVerifying Ollama connection...")
try:
    test_emb = get_embedding("test")
    print(f"  [OK] Ollama embeddings working (dim={len(test_emb)})")
except Exception as e:
    print(f"  [FAIL] Ollama embeddings FAILED: {e}")
    print("    Make sure Ollama is running: ollama serve")
    exit(1)


# ══════════════════════════════════════════════
#  RUN RAG PIPELINE AND EVALUATE
# ══════════════════════════════════════════════
def run_evaluation_for_namespace(namespace_name: str):
    results = []
    print(f"\n============================================================")
    print(f"Running Eval on Namespace: '{namespace_name}'")
    print(f"============================================================\n")

    for i, item in enumerate(test_data, 1):
        q = item["question"]
        gts = item["ground_truths"]
        print(f"  [{i}/{len(test_data)}] {q[:70]}...")

        try:
            rag_result = answer_query(q, namespace=namespace_name, index_name=ACTIVE_INDEX)
            answer = rag_result["answer"]
            retrieved_contexts = rag_result["retrieved_contexts"]

            print(f"      Retrieved {len(retrieved_contexts)} context chunks from Pinecone")

            # Answer quality metrics - Compare against all ground truth answers and get max
            rl = max((rouge_l(answer, gt) for gt in gts), default=0.0)

            # Semantic similarity - Compare against all ground truth answers and get max
            ss = max((semantic_similarity(answer, gt) for gt in gts), default=0.0)

            # Search/retrieval quality metrics — local Ollama (comparing against all GTs)
            ctx_sims = context_similarity(q, gts, retrieved_contexts)

            results.append({
                "question": q,
                "ground_truth": json.dumps(gts, ensure_ascii=False),
                "answer": answer,
                "num_contexts": len(retrieved_contexts),
                "rouge_l": rl,
                "semantic_similarity": ss,
                **ctx_sims,
            })

            print(f"      ROUGE-L={rl:.4f}  SemSim={ss:.4f}")
            print(f"      Context->Question: avg={ctx_sims['ctx_question_sim']:.4f}  best={ctx_sims['best_ctx_question_sim']:.4f}")
            print(f"      Context->GroundTr: avg={ctx_sims['ctx_ground_truth_sim']:.4f}  best={ctx_sims['best_ctx_gt_sim']:.4f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    if not results:
        print("\nNo questions processed. Ensure documents are ingested first!")
        return

    # ══════════════════════════════════════════════
    #  AGGREGATE AND DISPLAY RESULTS
    # ══════════════════════════════════════════════
    def safe_mean(values):
        clean = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
        return sum(clean) / len(clean) if clean else float("nan")

    avg_rl = safe_mean([r["rouge_l"] for r in results])
    avg_ss = safe_mean([r["semantic_similarity"] for r in results])
    avg_ctx_q = safe_mean([r["ctx_question_sim"] for r in results])
    avg_ctx_gt = safe_mean([r["ctx_ground_truth_sim"] for r in results])
    avg_best_ctx_q = safe_mean([r["best_ctx_question_sim"] for r in results])
    avg_best_ctx_gt = safe_mean([r["best_ctx_gt_sim"] for r in results])

    print("\n" + "=" * 60)
    print(f"   EVALUATION RESULTS - Namespace: {namespace_name}")
    print("=" * 60)
    print(f"  LLM                    : {settings.llm_model_name}")
    print(f"  Embedding model        : {settings.embedding_model_name}")
    print(f"  Questions evaluated    : {len(results)}")
    print(f"")
    print(f"  --- Answer Quality ---")
    print(f"  ROUGE-L                  : {avg_rl:.4f}")
    print(f"  Semantic Similarity      : {avg_ss:.4f}")
    print(f"")
    print(f"  --- Search/Retrieval Quality ---")
    print(f"  Context-Question (avg)   : {avg_ctx_q:.4f}")
    print(f"  Context-Question (best)  : {avg_best_ctx_q:.4f}")
    print(f"  Context-GroundTr (avg)   : {avg_ctx_gt:.4f}")
    print(f"  Context-GroundTr (best)  : {avg_best_ctx_gt:.4f}")
    print("=" * 60)

    # ──────────────────────────────────────────────
    # SAVE DETAILED RESULTS TO CSV
    # ──────────────────────────────────────────────
    csv_file = f"evaluation_results_{namespace_name}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question", "ground_truth", "answer", "num_contexts",
            "rouge_l", "semantic_similarity",
            "ctx_question_sim", "ctx_ground_truth_sim",
            "best_ctx_question_sim", "best_ctx_gt_sim"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDetailed results saved to {csv_file}")

    # Save a summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "namespace": namespace_name,
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
            "best_ctx_question_sim": round(float(avg_best_ctx_q), 4),
            "best_ctx_gt_sim": round(float(avg_best_ctx_gt), 4),
        }
    }

    sum_file = f"evaluation_summary_{namespace_name}.json"
    with open(sum_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {sum_file}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        namespaces = sys.argv[1:]
    else:
        # Default: evaluate the recursive strategy for the currently active model
        namespaces = [f"recursive-{ACTIVE_MODEL}"]

    print(f"  Active model  : {ACTIVE_MODEL}")
    print(f"  Target index  : {ACTIVE_INDEX}")
    print(f"  Namespaces    : {namespaces}\n")

    for ns in namespaces:
        run_evaluation_for_namespace(ns)
