import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import settings

# Ensure the data directory exists before SQLite tries to create the file
os.makedirs("data", exist_ok=True)

# connect_args is SQLite-specific: allows the same connection to be used
# across multiple threads (needed because FastAPI runs async workers)
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """
    FastAPI dependency that yields a DB session per request, then closes it.
    Usage in a router:  db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all tables if they don't exist. Called once at app startup."""
    from app.models import db_models  # noqa: F401 — ensures models are registered
    Base.metadata.create_all(bind=engine)
