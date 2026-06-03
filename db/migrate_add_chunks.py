# Run this once to add new columns to the existing table
# python db/migrate_add_chunks.py

from sqlalchemy import text
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.session import engine

columns_to_add = [
    ("retrieved_chunks", "TEXT"),
    ("ctx_q_sim", "FLOAT"),
    ("answer_sem_sim", "FLOAT"),
]

with engine.connect() as conn:
    for col_name, col_type in columns_to_add:
        try:
            conn.execute(text(f"ALTER TABLE autorag_query_log ADD COLUMN {col_name} {col_type}"))
            conn.commit()
            print(f"✅ Column '{col_name}' added to autorag_query_log")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                print(f"✅ Column '{col_name}' already exists — skipping")
                conn.rollback()
            else:
                print(f"❌ Error adding '{col_name}': {e}")
                conn.rollback()
