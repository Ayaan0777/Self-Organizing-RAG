"""
Clear database utility — for SQLite + Pinecone stack.

# Clear only SQLite logs, events, repairs (keep Pinecone vectors)
python db/clear_db.py --logs --confirm

# Clear only Pinecone vector embeddings (keep SQLite logs)
python db/clear_db.py --vectors --confirm

# Clear everything (Pinecone + SQLite)
python db/clear_db.py --all --confirm

# Interactive mode (asks for confirmation)
python db/clear_db.py --all
"""

import sys
import os
import argparse

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings


def clear_pinecone_vectors():
    """
    Clears all vectors from the Pinecone index for the configured namespace.
    """
    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)
        ns = settings.pinecone_namespace

        # Get index stats to show what we're deleting
        stats = index.describe_index_stats()
        ns_stats = stats.namespaces.get(ns, None)
        vector_count = ns_stats.vector_count if ns_stats else 0

        if vector_count == 0:
            print(f"  ⚠️ Namespace '{ns}' is already empty — nothing to delete.")
            return

        print(f"  🗑 Deleting {vector_count} vectors from namespace '{ns}'...")
        index.delete(delete_all=True, namespace=ns)
        print(f"  ✅ Pinecone namespace '{ns}' cleared!")

    except Exception as e:
        print(f"  ❌ Error clearing Pinecone: {e}")


def clear_sqlite_tables():
    """
    Clears all Auto-RAG monitoring tables in the local SQLite database.
    Includes the new Stage 2-4 tables (PipelineConfig, ChunkSnapshot, AdaptationLog).
    """
    import sqlite3

    db_path = os.path.join(os.path.dirname(__file__), "autorag.db")

    if not os.path.exists(db_path):
        print(f"  ⚠️ Database not found at {db_path} — nothing to clear.")
        return

    tables = [
        "autorag_repair_reports",
        "autorag_low_recall_events",
        "autorag_query_log",
        "autorag_eval_snapshots",
        "autorag_pipeline_config",
        "autorag_chunk_snapshots",
        "autorag_adaptation_log",
        "autorag_strategy_counters",
        "autorag_runtime_flags",
    ]

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        for table in tables:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            if cursor.fetchone():
                cursor.execute(f"DELETE FROM {table}")
                print(f"  🗑 Cleared {cursor.rowcount} rows from {table}")
            else:
                print(f"  ⚠️ Table {table} does not exist — skipping")

        conn.commit()
        conn.close()
        print("  ✅ SQLite tables cleared!")

    except Exception as e:
        print(f"  ❌ Error clearing SQLite: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clear the Auto-RAG database (SQLite + Pinecone)."
    )
    parser.add_argument(
        "--confirm", action="store_true", help="Skip the confirmation prompt."
    )
    parser.add_argument(
        "--vectors",
        action="store_true",
        help="Clear only Pinecone vector embeddings.",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Clear only SQLite tables (query logs, events, repairs, configs, snapshots).",
    )
    parser.add_argument(
        "--all", action="store_true", help="Clear both Pinecone vectors and SQLite tables."
    )

    args = parser.parse_args()

    # Default: if no specific flag, clear everything
    if not args.vectors and not args.logs and not getattr(args, "all"):
        do_vectors = True
        do_logs = True
    else:
        do_vectors = args.vectors or getattr(args, "all")
        do_logs = args.logs or getattr(args, "all")

    targets = []
    if do_vectors:
        targets.append("Pinecone vectors")
    if do_logs:
        targets.append("SQLite logs/events/repairs/configs")

    print(f"\n🎯 Targets: {', '.join(targets)}\n")

    if args.confirm:
        if do_vectors:
            print("── Clearing Pinecone ──")
            clear_pinecone_vectors()
        if do_logs:
            print("── Clearing SQLite ──")
            clear_sqlite_tables()
    else:
        response = input(
            f"⚠️  WARNING: This will permanently delete: {', '.join(targets)}.\n"
            f"   Are you sure? (y/N): "
        )
        if response.lower() == "y":
            if do_vectors:
                print("\n── Clearing Pinecone ──")
                clear_pinecone_vectors()
            if do_logs:
                print("\n── Clearing SQLite ──")
                clear_sqlite_tables()
        else:
            print("🛑 Operation cancelled.")

    print("\n✅ Done.\n")
