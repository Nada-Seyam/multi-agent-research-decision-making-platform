from __future__ import annotations

import sys
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

RAG_ROOT = Path(__file__).resolve().parents[1] / "Tools" / "RAG"
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

from retrieval import pipeline  # noqa: E402
from helper import helper as db_helper  # noqa: E402


def test_hybrid_retrieval_runs_dense_and_bm25_concurrently(monkeypatch):
    settings = SimpleNamespace(
        retrieval_enabled=True,
        retrieval_dense_top_n=10,
        retrieval_use_bm25=True,
        retrieval_bm25_top_n=10,
        retrieval_use_reranker=False,
        retrieval_reranker_top_n=10,
        retrieval_dedup_on_text=True,
        retrieval_max_per_document=2,
        retrieval_min_similarity=0.0,
        retrieval_rrf_k=60,
    )
    branch_barrier = Barrier(2)

    def dense_search(query_vector, top_n, filters=None):
        branch_barrier.wait(timeout=2)
        return []

    def bm25_search(query, top_n):
        branch_barrier.wait(timeout=2)
        return []

    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    monkeypatch.setattr(pipeline.dense_mod, "embed_query", lambda query: [0.1])
    monkeypatch.setattr(pipeline.dense_mod, "search_dense", dense_search)
    monkeypatch.setattr(pipeline.bm25_mod, "search_bm25", bm25_search)

    result = pipeline.run_search("latency test")

    assert result["dense_count"] == 0
    assert result["bm25_count"] == 0
    assert set(result["latency_ms"]) >= {"dense", "bm25", "total"}


def test_pooled_connection_is_rolled_back_and_returned(monkeypatch):
    class FakeConnection:
        closed = False
        rollback_count = 0

        def rollback(self):
            self.rollback_count += 1

    connection = FakeConnection()

    class FakePool:
        returned = []

        def getconn(self):
            return connection

        def putconn(self, item, *, close):
            self.returned.append((item, close))

    pool = FakePool()
    monkeypatch.setattr(db_helper, "_CONNECTION_POOL", pool)

    with db_helper.pooled_connection() as borrowed:
        assert borrowed is connection

    assert connection.rollback_count == 1
    assert pool.returned == [(connection, False)]
