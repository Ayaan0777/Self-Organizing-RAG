"""
Database session factory for Auto-RAG.
Uses SQLite stored at db/autorag.db alongside this file.
Tables are auto-created on first import.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base

# Store the DB file next to this module
_DB_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_PATH = os.path.join(_DB_DIR, "autorag.db")
_DATABASE_URL = f"sqlite:///{_DB_PATH}"

engine = create_engine(_DATABASE_URL, echo=False)
Base.metadata.create_all(engine)

_SessionFactory = sessionmaker(bind=engine)


def get_session():
    """Return a new SQLAlchemy session. Caller is responsible for closing it."""
    return _SessionFactory()
