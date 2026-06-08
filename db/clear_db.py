"""

# Clear only query logs, events, repairs (keep embeddings)
python clear_db.py --logs --confirm

# Clear only vector embeddings (keep logs)
python clear_db.py --vectors --confirm

# Clear everything (vectors + logs)
python clear_db.py --all --confirm

# Interactive mode (asks for confirmation)
python clear_db.py --logs

"""


import argparse
import psycopg
from config import settings

def clear_vector_db():
    """
    Connects to the PostgreSQL database and safely clears all entries from the PGVector collection.
    """
    
    db_url = settings.database_url
    collection_name = settings.collection_name
    
    # We strip out the SQLAlchemy specific prefix 'postgresql+psycopg://' and replacing
    # it with the standard 'postgresql://' so the raw psycopg library can parse it.
    if db_url.startswith("postgresql+psycopg://"):
        raw_db_url = db_url.replace("postgresql+psycopg://", "postgresql://")
    else:
        raw_db_url = db_url
        
    try:
        # Connect to the Postgres database
        print(f"🔌 Connecting to database to clear collection: '{collection_name}'...")
        with psycopg.connect(raw_db_url) as conn:
            with conn.cursor() as cur:
                
                # Check if the langchain_pg_embedding table exists
                cur.execute(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'langchain_pg_embedding');"
                )
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    print("⚠️ The PGVector tables do not exist yet. There is nothing to clear!")
                    return
                
                # We need to find the UUID of our specific collection first
                cur.execute(
                    "SELECT uuid FROM langchain_pg_collection WHERE name = %s;", 
                    (collection_name,)
                )
                
                result = cur.fetchone()
                
                if not result:
                    print(f"⚠️ Collection '{collection_name}' not found. It might already be empty.")
                    return
                
                collection_uuid = result[0]
                
                # Delete all embeddings that belong to this collection
                cur.execute(
                    "DELETE FROM langchain_pg_embedding WHERE collection_id = %s;",
                    (collection_uuid,)
                )
                
                # Get the number of deleted rows
                deleted_rows = cur.rowcount
                
                conn.commit()
                print(f"✅ Success! Deleted {deleted_rows} embedded chunks from '{collection_name}'.")
                
    except Exception as e:
        print(f"❌ Error clearing the database: {e}")

def clear_autorag_tables():
    """
    Clears all Auto-RAG monitoring tables:
    autorag_query_log, autorag_low_recall_events, autorag_repair_reports, autorag_eval_snapshots
    """
    db_url = settings.database_url
    if db_url.startswith("postgresql+psycopg://"):
        raw_db_url = db_url.replace("postgresql+psycopg://", "postgresql://")
    else:
        raw_db_url = db_url

    tables = [
        "autorag_repair_reports",
        "autorag_low_recall_events",
        "autorag_query_log",
        "autorag_eval_snapshots",
    ]

    try:
        print("🔌 Connecting to database to clear Auto-RAG tables...")
        with psycopg.connect(raw_db_url) as conn:
            with conn.cursor() as cur:
                for table in tables:
                    cur.execute(
                        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s);",
                        (table,)
                    )
                    if cur.fetchone()[0]:
                        cur.execute(f"DELETE FROM {table};")
                        print(f"  🗑 Cleared {cur.rowcount} rows from {table}")
                    else:
                        print(f"  ⚠️ Table {table} does not exist — skipping")
                conn.commit()
        print("✅ Auto-RAG tables cleared!")
    except Exception as e:
        print(f"❌ Error clearing Auto-RAG tables: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean the local RAG database.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    parser.add_argument(
        "--vectors",
        action="store_true",
        help="Clear only vector embeddings (langchain_pg_embedding).",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Clear only Auto-RAG tables (query logs, events, repairs, eval snapshots).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clear both vectors and Auto-RAG tables.",
    )

    args = parser.parse_args()

    # Default: if no specific flag, clear everything
    if not args.vectors and not args.logs and not getattr(args, 'all'):
        do_vectors = True
        do_logs = True
    else:
        do_vectors = args.vectors or getattr(args, 'all')
        do_logs = args.logs or getattr(args, 'all')

    targets = []
    if do_vectors:
        targets.append("vector embeddings")
    if do_logs:
        targets.append("Auto-RAG logs/events/repairs")

    if args.confirm:
        if do_vectors:
            clear_vector_db()
        if do_logs:
            clear_autorag_tables()
    else:
        response = input(
            f"⚠️ WARNING: This will permanently delete: {', '.join(targets)}.\n"
            f"   Are you sure? (y/N): "
        )
        if response.lower() == 'y':
            if do_vectors:
                clear_vector_db()
            if do_logs:
                clear_autorag_tables()
        else:
            print("🛑 Operation cancelled.")
