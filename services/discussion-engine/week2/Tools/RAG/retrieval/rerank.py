"""Cross-encoder reranker.

Wraps a ``sentence_transformers.CrossEncoder`` so the API surface is
``(query, candidate_list) -> re-ranked candidate_list``. The model is
loaded once and cached for the process lifetime; on a CPU-only box the
first call pays the load cost (a few seconds for MiniLM, ~30s for
``bge-reranker-v2-m3``) and subsequent calls are sub-second for ~50
candidates.

The default model is ``cross-encoder/ms-marco-MiniLM-L-6-v2``: ~22M
parameters, ~90 MB on disk, runs comfortably on CPU. Switch to
``BAAI/bge-reranker-v2-m3`` if you have a GPU and want stronger quality.
"""
from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

from helper import get_settings

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        settings = get_settings()
        device = settings.retrieval_device.strip().lower()
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = CrossEncoder(settings.retrieval_reranker_model, device=device)
    return _model


def reset_model_for_tests() -> None:
    """Drop the cached reranker for isolated test fixtures."""
    global _model
    _model = None


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Return ``candidates`` re-ordered by cross-encoder score (descending).

    Each result gets a ``rerank_score`` field. The original ranking is
    preserved as ``original_rank`` so we can build diagnostics.

    If ``top_n`` is given, only the first ``top_n`` items are returned
    (post-rerank). If ``candidates`` is empty, the input is returned
    unchanged.
    """
    if not candidates:
        return list(candidates)

    import torch
    model = _get_model()
    pairs = [(query, c.get("embedding_text") or "") for c in candidates]
    
    with torch.inference_mode():
        # Using batch_size=8 allows sentence-transformers to sort inputs by length internally,
        # drastically reducing padding overhead compared to processing all candidates in one batch.
        scores = model.predict(pairs, batch_size=8, show_progress_bar=False)

    enriched: list[dict[str, Any]] = []
    for i, (c, s) in enumerate(zip(candidates, scores)):
        row = dict(c)
        row["original_rank"] = i + 1
        row["rerank_score"] = float(s)
        enriched.append(row)

    enriched.sort(key=lambda r: r["rerank_score"], reverse=True)
    if top_n is not None and top_n > 0:
        enriched = enriched[:top_n]
    return enriched
