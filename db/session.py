import os
import logging
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from db.models import Base

# This dynamically finds the absolute path of the 'db' folder (where this file lives)
# and forces the database to always be created/read from exactly here.
_db_dir = os.path.dirname(os.path.abspath(__file__))
_db_path = os.path.join(_db_dir, "autorag.db")
DATABASE_URL = f"sqlite:///{_db_path}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def _migrate_repair_reports():
    """
    Adds new SH2 columns to autorag_repair_reports if they don't exist.
    SQLAlchemy's create_all() only creates NEW tables, not new columns.
    This handles the migration for existing databases.
    """
    insp = inspect(engine)
    if "autorag_repair_reports" not in insp.get_table_names():
        return  # table doesn't exist yet, create_all() will handle it

    existing_cols = {c["name"] for c in insp.get_columns("autorag_repair_reports")}
    migrations = {
        "chunk_size_used": "ALTER TABLE autorag_repair_reports ADD COLUMN chunk_size_used INTEGER",
        "repair_reason":   "ALTER TABLE autorag_repair_reports ADD COLUMN repair_reason VARCHAR(100)",
        "rolled_back":     "ALTER TABLE autorag_repair_reports ADD COLUMN rolled_back BOOLEAN DEFAULT 0",
    }
    with engine.begin() as conn:
        for col_name, ddl in migrations.items():
            if col_name not in existing_cols:
                conn.execute(text(ddl))
                logging.info(f"[db] migrated: added column '{col_name}' to autorag_repair_reports")


def init_db():
    """
    Creates all autorag_* tables in the local SQLite DB.
    Called once at FastAPI startup. Safe to call repeatedly — skips existing tables.
    Also runs migrations to add new columns to existing tables.
    """
    Base.metadata.create_all(bind=engine)
    _migrate_repair_reports()


def get_session():
    """Returns a new SQLAlchemy session. Caller is responsible for closing it."""
    return SessionLocal()