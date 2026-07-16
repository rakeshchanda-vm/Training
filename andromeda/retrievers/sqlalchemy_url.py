"""Helpers for SQLAlchemy engine URLs used by retriever storage."""

from __future__ import annotations


def sqlalchemy_async_postgres_url(connection_string: str) -> str:
    """
    Return a URL suitable for ``create_async_engine`` against Postgres.

    Bare ``postgresql://`` / ``postgres://`` URLs select a synchronous driver by
    default (often psycopg2), which cannot be used with SQLAlchemy's asyncio
    layer. Rewrite those (and other sync-only Postgres dialects) to
    ``postgresql+psycopg_async`` so ``andromeda[retrievers-postgres]`` (psycopg3)
    can serve async paths without requiring asyncpg.
    """
    try:
        from sqlalchemy.engine.url import make_url
    except ImportError:
        return connection_string
    try:
        url = make_url(connection_string)
    except Exception:
        return connection_string
    driver = url.drivername
    if driver in ("postgresql+asyncpg", "postgresql+psycopg_async"):
        return connection_string
    if driver in ("postgresql", "postgres", "postgresql+psycopg2", "postgresql+psycopg"):
        return str(url.set(drivername="postgresql+psycopg_async"))
    return connection_string
