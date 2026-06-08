"""
Auto Indexing Engine v1 — Month 4 Deliverable
==============================================
Detects stale/drifted embeddings and refreshes them without
wiping the entire index.

Components:
  1. Staleness Detector   — finds chunks whose embeddings have drifted
  2. Partial Re-embedder  — re-embeds only the stale chunks (upsert)
  3. Index Refresher      — upsert new + delete orphaned vectors
  4. Consistency Checker   — verifies index health and metadata integrity

Usage:
    from auto_indexer import AutoIndexer

    indexer = AutoIndexer(namespace="mxbai-embed-large")
    report  = indexer.run_full_refresh()
    stale   = indexer.detect_stale_chunks(sample_size=50)
    health  = indexer.check_consistency()
"""
import json
import time
import numpy as np
from datetime import datetime
from typing import Optional

from config import settings
from services.llm_factory import get_embeddings, get_pinecone_index, get_vector_store


def _cosine_sim(a, b) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


class AutoIndexer:
    """
    Auto Indexing Engine that detects and repairs embedding staleness
    without destroying the existing index.
    """

    def __init__(self, namespace: str = None):
        self.namespace = namespace or settings.pinecone_namespace
        self.index = get_pinecone_index()
        self.embeddings = get_embeddings()
        self.vs = get_vector_store(self.namespace)

    # ──────────────────────────────────────────────────────────
    #  1. STALENESS DETECTOR
    # ──────────────────────────────────────────────────────────

    def detect_stale_chunks(
        self,
        sample_size: int = 50,
        drift_threshold: float = 0.95,
    ) -> dict:
        """
        Detects chunks whose stored embeddings have drifted from
        what the current embedding model would produce.

        How it works:
          1. Sample random vectors from the namespace
          2. For each: re-embed the text with the current model
          3. Compare stored vs fresh embedding (cosine similarity)
          4. If similarity < drift_threshold → chunk is stale

        Args:
            sample_size: Number of random vectors to sample.
            drift_threshold: Cosine sim below this = stale.

        Returns:
            {
                "total_sampled": int,
                "stale_count": int,
                "stale_ids": list[str],
                "avg_drift": float,
                "worst_drift": float,
            }
        """
        print(f"[auto-indexer] Sampling {sample_size} vectors for staleness check...")

        # Use a random query to get sample vectors
        # Pinecone doesn't support random sampling, so we use diverse probe queries
        probe_queries = [
            "algorithm", "data structure", "network", "database",
            "function", "variable", "class", "object", "array",
            "system", "process", "memory", "security", "protocol",
        ]

        seen_ids = set()
        samples = []

        for probe in probe_queries:
            if len(samples) >= sample_size:
                break
            probe_emb = self.embeddings.embed_query(probe)
            results = self.index.query(
                vector=probe_emb,
                top_k=min(10, sample_size - len(samples) + 5),
                namespace=self.namespace,
                include_metadata=True,
                include_values=True,
            )
            for m in results.matches:
                if m.id not in seen_ids and len(samples) < sample_size:
                    seen_ids.add(m.id)
                    samples.append(m)

        if not samples:
            return {"total_sampled": 0, "stale_count": 0, "stale_ids": [],
                    "avg_drift": 0.0, "worst_drift": 0.0}

        stale_ids = []
        drifts = []

        for m in samples:
            text = m.metadata.get("text", "")
            if not text:
                continue

            # Re-embed with current model
            fresh_emb = np.array(self.embeddings.embed_query(text[:500]))
            stored_emb = np.array(m.values)

            sim = _cosine_sim(fresh_emb, stored_emb)
            drift = 1.0 - sim
            drifts.append(drift)

            if sim < drift_threshold:
                stale_ids.append(m.id)

        avg_drift = float(np.mean(drifts)) if drifts else 0.0
        worst_drift = float(max(drifts)) if drifts else 0.0

        print(f"[auto-indexer] Sampled {len(samples)}, "
              f"stale={len(stale_ids)}, avg_drift={avg_drift:.4f}")

        return {
            "total_sampled": len(samples),
            "stale_count": len(stale_ids),
            "stale_ids": stale_ids,
            "avg_drift": round(avg_drift, 4),
            "worst_drift": round(worst_drift, 4),
        }

    # ──────────────────────────────────────────────────────────
    #  2. PARTIAL RE-EMBEDDER
    # ──────────────────────────────────────────────────────────

    def reembed_stale(self, stale_ids: list[str]) -> dict:
        """
        Re-embeds only the specified stale vectors.
        Uses Pinecone upsert to update in-place (no delete needed).

        Steps:
          1. Fetch the stale vectors (with metadata + text)
          2. Re-embed the text with the current model
          3. Upsert the updated vectors back to Pinecone

        Args:
            stale_ids: List of Pinecone vector IDs to re-embed.

        Returns:
            {"refreshed": int, "failed": int, "ids": list[str]}
        """
        if not stale_ids:
            return {"refreshed": 0, "failed": 0, "ids": []}

        print(f"[auto-indexer] Re-embedding {len(stale_ids)} stale vectors...")

        refreshed = 0
        failed = 0
        refreshed_ids = []

        # Process in batches of 50
        for i in range(0, len(stale_ids), 50):
            batch_ids = stale_ids[i:i + 50]

            # Fetch current vectors
            fetch_result = self.index.fetch(ids=batch_ids, namespace=self.namespace)

            upsert_batch = []
            for vid, vec_data in fetch_result.vectors.items():
                text = vec_data.metadata.get("text", "")
                if not text:
                    failed += 1
                    continue

                try:
                    fresh_emb = self.embeddings.embed_query(text[:500])
                    upsert_batch.append({
                        "id": vid,
                        "values": fresh_emb,
                        "metadata": vec_data.metadata,
                    })
                    refreshed_ids.append(vid)
                    refreshed += 1
                except Exception as e:
                    print(f"[auto-indexer] Failed to re-embed {vid}: {e}")
                    failed += 1

            # Upsert updated vectors
            if upsert_batch:
                self.index.upsert(
                    vectors=upsert_batch,
                    namespace=self.namespace,
                )

        print(f"[auto-indexer] Refreshed {refreshed}, failed {failed}")
        return {"refreshed": refreshed, "failed": failed, "ids": refreshed_ids}

    # ──────────────────────────────────────────────────────────
    #  3. INDEX REFRESH (upsert new + delete orphaned)
    # ──────────────────────────────────────────────────────────

    def refresh_index(
        self,
        documents: list = None,
        source: str = None,
    ) -> dict:
        """
        Partial index refresh: upserts new/changed chunks and removes
        orphaned vectors that no longer match any document.

        If documents are provided, chunks them and upserts.
        If source is provided, checks for orphaned vectors from that source.

        Args:
            documents: New LangChain Documents to upsert.
            source: Source identifier to check for orphans.

        Returns:
            {"upserted": int, "orphans_deleted": int}
        """
        upserted = 0
        orphans_deleted = 0

        # Upsert new documents
        if documents:
            print(f"[auto-indexer] Upserting {len(documents)} documents...")
            self.vs.add_documents(documents)
            upserted = len(documents)

        # Find and delete orphaned vectors (vectors with no matching text)
        if source:
            print(f"[auto-indexer] Checking for orphaned vectors from '{source}'...")
            probe_emb = self.embeddings.embed_query(f"content from {source}")
            results = self.index.query(
                vector=probe_emb,
                top_k=100,
                namespace=self.namespace,
                filter={"source": source},
                include_metadata=True,
            )

            orphan_ids = []
            for m in results.matches:
                text = m.metadata.get("text", "")
                if not text or len(text.strip()) < 10:
                    orphan_ids.append(m.id)

            if orphan_ids:
                self.index.delete(ids=orphan_ids, namespace=self.namespace)
                orphans_deleted = len(orphan_ids)
                print(f"[auto-indexer] Deleted {orphans_deleted} orphaned vectors")

        return {"upserted": upserted, "orphans_deleted": orphans_deleted}

    # ──────────────────────────────────────────────────────────
    #  4. CONSISTENCY CHECKER
    # ──────────────────────────────────────────────────────────

    def check_consistency(self) -> dict:
        """
        Runs index health checks:
          - Total vector count
          - Dimension check
          - Metadata integrity (% of vectors with text metadata)
          - Empty text detection (vectors with missing/empty text)
          - Score distribution check (are most vectors retrievable?)

        Returns a health report dict.
        """
        print(f"[auto-indexer] Running consistency check on '{self.namespace}'...")

        stats = self.index.describe_index_stats()
        ns_stats = stats.namespaces.get(self.namespace)

        total_vectors = ns_stats.vector_count if ns_stats else 0
        dimension = stats.dimension

        # Sample vectors to check metadata integrity
        probe_queries = ["data", "algorithm", "system", "function", "network"]
        seen_ids = set()
        has_text = 0
        empty_text = 0
        no_metadata = 0
        total_checked = 0

        for probe in probe_queries:
            probe_emb = self.embeddings.embed_query(probe)
            results = self.index.query(
                vector=probe_emb,
                top_k=20,
                namespace=self.namespace,
                include_metadata=True,
            )
            for m in results.matches:
                if m.id in seen_ids:
                    continue
                seen_ids.add(m.id)
                total_checked += 1

                if not m.metadata:
                    no_metadata += 1
                elif m.metadata.get("text", "").strip():
                    has_text += 1
                else:
                    empty_text += 1

        metadata_integrity = round(has_text / total_checked * 100, 1) if total_checked else 0

        # Score distribution: check if a generic query returns reasonable scores
        test_emb = self.embeddings.embed_query("common concept")
        score_results = self.index.query(
            vector=test_emb,
            top_k=10,
            namespace=self.namespace,
        )
        scores = [m.score for m in score_results.matches] if score_results.matches else []
        avg_score = round(float(np.mean(scores)), 4) if scores else 0.0

        health = "HEALTHY" if metadata_integrity > 90 and total_vectors > 0 else "DEGRADED"
        if total_vectors == 0:
            health = "EMPTY"

        report = {
            "status": health,
            "namespace": self.namespace,
            "total_vectors": total_vectors,
            "dimension": dimension,
            "sampled": total_checked,
            "metadata_integrity_pct": metadata_integrity,
            "has_text": has_text,
            "empty_text": empty_text,
            "no_metadata": no_metadata,
            "avg_retrieval_score": avg_score,
            "timestamp": datetime.utcnow().isoformat(),
        }

        print(f"[auto-indexer] Health: {health} | vectors={total_vectors} "
              f"| integrity={metadata_integrity}% | avg_score={avg_score}")

        return report

    # ──────────────────────────────────────────────────────────
    #  FULL PIPELINE
    # ──────────────────────────────────────────────────────────

    def run_full_refresh(
        self,
        sample_size: int = 50,
        drift_threshold: float = 0.95,
        auto_fix: bool = True,
    ) -> dict:
        """
        Runs the complete auto-indexing pipeline:
          1. Consistency check
          2. Staleness detection
          3. Auto re-embedding (if auto_fix=True)
          4. Final consistency check

        Returns a full report.
        """
        t0 = time.time()
        print("\n" + "=" * 60)
        print("  AUTO-INDEXER — Full Refresh Pipeline")
        print("=" * 60)

        # Step 1: Initial health check
        health_before = self.check_consistency()

        # Step 2: Detect stale chunks
        staleness = self.detect_stale_chunks(
            sample_size=sample_size,
            drift_threshold=drift_threshold,
        )

        # Step 3: Re-embed stale chunks
        reembed_result = {"refreshed": 0, "failed": 0}
        if auto_fix and staleness["stale_ids"]:
            reembed_result = self.reembed_stale(staleness["stale_ids"])

        # Step 4: Final health check
        health_after = self.check_consistency()

        duration_ms = int((time.time() - t0) * 1000)

        report = {
            "health_before": health_before,
            "staleness": staleness,
            "reembedded": reembed_result,
            "health_after": health_after,
            "duration_ms": duration_ms,
        }

        print(f"\n[auto-indexer] Full refresh complete in {duration_ms}ms")
        print("=" * 60)

        # Persist to DB
        try:
            from db.session import get_session
            from db.models import EvalSnapshot
            s = get_session()
            snap = EvalSnapshot(
                namespace=self.namespace,
                llm="auto-indexer",
                embeddings=settings.embedding_model_name,
                rouge_l=0.0,
                sem_sim=0.0,
                ctx_q_sim=staleness["avg_drift"],
                ctx_gt_sim=float(staleness["stale_count"]),
            )
            s.add(snap)
            s.commit()
            s.close()
        except Exception:
            pass

        return report
