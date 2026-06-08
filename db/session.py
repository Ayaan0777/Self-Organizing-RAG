import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base

# This dynamically finds the absolute path of the 'db' folder (where this file lives)
# and forces the database to always be created/read from exactly here.
_db_dir = os.path.dirname(os.path.abspath(__file__))
_db_path = os.path.join(_db_dir, "autorag.db")
DATABASE_URL = f"sqlite:///{_db_path}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def init_db():
    """
    Creates all autorag_* tables in the local SQLite DB.
    Called once at FastAPI startup. Safe to call repeatedly — skips existing tables.
    """
    Base.metadata.create_all(bind=engine)

def get_session():
    """Returns a new SQLAlchemy session. Caller is responsible for closing it."""
    return SessionLocal()