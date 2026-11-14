from __future__ import annotations

import sys
from pathlib import Path


RAG_ROOT = Path(__file__).resolve().parents[1] / "Tools" / "RAG"
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

from retrieval.filters import apply_metadata_filters  # noqa: E402


def test_query_groups_filter_bm25_rows_by_document_topics():
    candidates = [
        {
            "chunk_id": "matching",
            "bm25_score": 2.0,
            "document_topics": ["architecture", "latency"],
        },
        {
            "chunk_id": "wrong-group",
            "bm25_score": 3.0,
            "document_topics": ["fine-tuning"],
        },
        {
            "chunk_id": "missing-metadata",
            "bm25_score": 4.0,
        },
    ]

    filtered = apply_metadata_filters(
        candidates,
        {"query_groups": ["architecture"]},
    )

    assert [row["chunk_id"] for row in filtered] == ["matching"]


def test_query_groups_keep_dense_rows_already_filtered_in_sql():
    dense_row = {"chunk_id": "dense", "similarity": 0.9}

    assert apply_metadata_filters(
        [dense_row],
        {"query_groups": ["architecture"]},
    ) == [dense_row]
