from __future__ import annotations

import atexit
from threading import Lock
from typing import Any

import httpx

from src.helper.helper import get_settings

REQUEST_TIMEOUT_SECONDS = 60.0
MAX_ERROR_BODY_CHARS = 500
_CLIENT: httpx.Client | None = None
_CLIENT_LOCK = Lock()


def _get_client() -> httpx.Client:
    """Return one thread-safe client so concurrent queries reuse connections."""
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = httpx.Client(
                    timeout=REQUEST_TIMEOUT_SECONDS,
                    limits=httpx.Limits(
                        max_connections=16,
                        max_keepalive_connections=8,
                    ),
                )
    return _CLIENT


def close_client() -> None:
    """Release pooled sockets during process shutdown and in isolated tests."""
    global _CLIENT
    with _CLIENT_LOCK:
        client, _CLIENT = _CLIENT, None
    if client is not None:
        client.close()


atexit.register(close_client)


def _get_base_url() -> str:
    base = get_settings().RETRIEVAL_API_BASE_URL
    if not base:
        raise RuntimeError(
            "RETRIEVAL_API_BASE_URL is not set. "
            "Set it in src/.env (e.g. RETRIEVAL_API_BASE_URL=http://localhost:8000) "
            "or export it in the shell."
        )
    return base.rstrip("/")


def _get_top_k() -> int:
    raw = get_settings().RETRIEVAL_TOP_K
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"RETRIEVAL_TOP_K must be an integer, got {raw!r}") from exc


def retrieve(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Call POST /search and return the list of ChunkResult dicts.

    Raises RuntimeError on transport / non-2xx errors so the caller sees
    the failure rather than silently receiving empty chunks.
    """
    if not query or not query.strip():
        raise RuntimeError("retrieve(): query must be a non-empty string")

    url = f"{_get_base_url()}/search"
    payload = {"query": query, "top_k": top_k if top_k is not None else _get_top_k()}

    try:
        response = _get_client().post(url, json=payload)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"RAG /search transport error to {url}: {exc!r}") from exc

    if response.status_code >= 400:
        body = response.text[:MAX_ERROR_BODY_CHARS] if response.text else "<empty body>"
        raise RuntimeError(
            f"RAG /search failed: HTTP {response.status_code} from {url} — {body}"
        )

    try:
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"RAG /search returned non-JSON body from {url}: {exc!r}") from exc

    results = data.get("results")
    if not isinstance(results, list):
        raise RuntimeError(
            f"RAG /search response missing 'results' list (got {type(results).__name__})"
        )
    return results
