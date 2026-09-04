from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from .models import Base

_settings = get_settings()
_settings.ensure_dirs()

engine = create_engine(
    _settings.sqlite_url,
    # SQLite by default refuses use from a thread other than the one
    # that opened it. FastAPI serves requests from a thread pool, so we
    # disable that check - safe here because SQLAlchemy's connection
    # pool hands each thread its own connection.
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create tables that don't exist yet. Called once at startup.

    Fine for a solo local project. If the schema ever needs to change
    *without losing data*, that's the point to bring in Alembic
    migrations - create_all() never alters an existing table.
    """
    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    """FastAPI dependency. Use as: db: Session = Depends(get_db)

    Yields a session and guarantees it's closed once the response is
    sent, even if the handler raised.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session for use outside FastAPI (scripts, ingestion).

    Commits on success, rolls back on any exception. Use this in the
    CLI ingestion path so a crash halfway through can't leave the
    metadata table describing documents that were never indexed.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
