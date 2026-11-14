"""Centralized project settings loaded from .env via pydantic-settings.

All environment variables are optional — the project runs with sensible
defaults that match the values previously hardcoded in settings.yaml.
A ``.env`` file at the repo root overrides any default.

Usage::

    from src.helper.helper import get_settings

    settings = get_settings()
    print(settings.request_delay_seconds)   # 1.0 (or whatever .env says)
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Iterator

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from pydantic_settings import BaseSettings, SettingsConfigDict

_CONNECTION_POOL: ThreadedConnectionPool | None = None
_POOL_LOCK = Lock()

def get_connection():
    """Open a fresh psycopg2 connection using the configured Postgres creds.

    Lives here (rather than in any one caller) so the loader, dense
    retriever, BM25 indexer, and rerankers all reach the database through
    the same code path. The connection is *not* pooled — every caller
    opens, uses, and closes its own. For the 123-document corpus this is
    the simplest correct option.
    """
    settings = get_settings()
    return psycopg2.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        dbname=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        sslmode=settings.DB_SSLMODE
    )


def _get_connection_pool() -> ThreadedConnectionPool:
    global _CONNECTION_POOL
    if _CONNECTION_POOL is None:
        with _POOL_LOCK:
            if _CONNECTION_POOL is None:
                settings = get_settings()
                _CONNECTION_POOL = ThreadedConnectionPool(
                    minconn=settings.DB_POOL_MIN_SIZE,
                    maxconn=settings.DB_POOL_MAX_SIZE,
                    host=settings.DB_HOST,
                    port=settings.DB_PORT,
                    dbname=settings.DB_NAME,
                    user=settings.DB_USER,
                    password=settings.DB_PASSWORD,
                    sslmode=settings.DB_SSLMODE,
                )
    return _CONNECTION_POOL


@contextmanager
def pooled_connection() -> Iterator[object]:
    """Borrow a database connection and safely return it to the shared pool."""
    pool = _get_connection_pool()
    connection = pool.getconn()
    discard = False
    try:
        yield connection
        connection.rollback()
    except Exception:
        discard = True
        raise
    finally:
        pool.putconn(connection, close=discard or bool(connection.closed))


def close_connection_pool() -> None:
    global _CONNECTION_POOL
    with _POOL_LOCK:
        pool, _CONNECTION_POOL = _CONNECTION_POOL, None
    if pool is not None:
        pool.closeall()


atexit.register(close_connection_pool)


class Settings(BaseSettings):
    """Project-wide configuration.

    Field names map 1-to-1 to env-var names (uppercased).
    e.g.  ``request_delay_seconds``  ↔  ``REQUEST_DELAY_SECONDS``
    """

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    DB_SSLMODE: str = "require"
    DB_POOL_MIN_SIZE: int = 1
    DB_POOL_MAX_SIZE: int = 8
    model_name: str = "BAAI/bge-m3"
    retrieval_device: str = "auto"
    retrieval_enabled: bool = True

    # Dense retrieval: candidate pool size fed into the next stage.
    retrieval_dense_top_n: int = 40

    # Optional BM25 lexical retrieval. Off by default; flip on for E/F runs.
    retrieval_use_bm25: bool = True
    retrieval_bm25_top_n: int = 40
    # Reciprocal Rank Fusion constant. Standard value is 60.
    retrieval_rrf_k: int = 60

    # Result-level post-filters. Applied after candidate pool construction
    # (dense + optional BM25 fused) and before reranking.
    retrieval_dedup_on_text: bool = True
    retrieval_max_per_document: int = 2
    retrieval_min_similarity: float = 0.0

    # Optional reranker. Off by default; flip on for D/F runs.
    retrieval_use_reranker: bool = True
    retrieval_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_reranker_top_n: int = 20  

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton so the .env file is read only once."""
    return Settings()
