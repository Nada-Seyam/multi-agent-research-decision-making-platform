"""Small, centralized adapter around the existing Week 2 framework.

Week 2 uses ``src`` as its top-level package and expects its project root to
be on ``sys.path``. Only this module handles that layout detail; the rest of
Week 3 works with the Week 2 public functions through these wrappers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from .config import PROJECT_ROOT


class Week2IntegrationError(RuntimeError):
    """Raised when the Week 2 framework cannot be loaded from the workspace."""


_ACTIVE_WEEK2_ROOT: Path | None = None


def resolve_week2_root(week2_root: str | Path | None = None) -> Path:
    root = Path(week2_root) if week2_root is not None else PROJECT_ROOT / "week2"
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    if not (root / "src" / "agents").is_dir():
        raise Week2IntegrationError(
            f"Week 2 agents package was not found under {root!s}. "
            "Set DISCUSSION_WEEK2_ROOT to the correct workspace-relative path."
        )
    return root


def _prepare_import_path(week2_root: str | Path | None = None) -> Path:
    global _ACTIVE_WEEK2_ROOT
    if week2_root is None and _ACTIVE_WEEK2_ROOT is not None:
        week2_root = _ACTIVE_WEEK2_ROOT
    root = resolve_week2_root(week2_root)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    _ACTIVE_WEEK2_ROOT = root
    return root


def load_week2_personas(
    week2_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load the Week 2 persona mapping without duplicating its configuration."""
    _prepare_import_path(week2_root)
    from src.agents.config import get_personas

    data = get_personas()
    personas = data.get("personas", {}) if isinstance(data, dict) else {}
    personas = {
        persona_id: persona
        for persona_id, persona in personas.items()
        if persona_id != "shared_rules" and isinstance(persona, dict)
    }
    if not isinstance(personas, dict) or not personas:
        raise Week2IntegrationError("Week 2 Personas.yaml contains no personas.")
    return personas


def load_objective(path: str | Path, week2_root: str | Path | None = None) -> Any:
    _prepare_import_path(week2_root)
    from src.agents.objective import load_objective as week2_load_objective

    return week2_load_objective(path)


def plan_research(
    objective: Any,
    memory_summary: dict[str, Any] | None = None,
    query_limit: int = 5,
) -> Any:
    _prepare_import_path()
    from src.agents.planner import plan_research as week2_plan_research

    return week2_plan_research(
        objective,
        memory_summary=memory_summary,
        query_limit=query_limit,
    )


def build_evidence_pool(
    queries: list[str],
    objective: Any | None = None,
    *,
    progress_callback: Callable[[Any, int, int], None] | None = None,
) -> list[Any]:
    _prepare_import_path()
    from src.agents.evidence_pool import (
        build_evidence_pool as week2_build_evidence_pool,
    )

    return week2_build_evidence_pool(
        queries,
        objective=objective,
        progress_callback=progress_callback,
    )


def consolidate_evidence(pool: list[Any], objective: Any) -> Any:
    _prepare_import_path()
    from src.agents.consolidator import (
        consolidate_evidence as week2_consolidate_evidence,
    )

    return week2_consolidate_evidence(pool, objective)


def run_for_persona(
    persona_id: str,
    objective: Any,
    dossier: Any,
    *,
    memory_context: str | None = None,
) -> dict[str, Any]:
    _prepare_import_path()
    from src.agents.graph import run_for_persona as week2_run_for_persona

    return week2_run_for_persona(
        persona_id,
        objective,
        dossier,
        memory_context=memory_context,
    )


def generate_text(system_prompt: str, user_prompt: str) -> str:
    """Generate text with the Week 2 configured model client."""
    _prepare_import_path()
    from src.agents.llm import chat

    return chat(system_prompt, user_prompt)


def ensure_local_rag() -> bool:
    """Ensure the configured local Week 2 RAG API is ready.

    This blocking entry point remains available for CLI and external callers.
    The FastAPI lifespan uses :func:`start_local_rag` so startup stays
    non-blocking while readiness is exposed through the health endpoint.
    """
    _prepare_import_path()
    from src.agents.run import _ensure_rag_up

    return _ensure_rag_up()


def start_local_rag() -> bool:
    """Start the configured local RAG API without waiting for model warm-up."""
    _prepare_import_path()
    from src.agents.run import _start_rag_if_needed

    return _start_rag_if_needed()


def stop_local_rag() -> None:
    """Stop the RAG child started by either local startup entry point."""
    _prepare_import_path()
    from src.agents.run import _teardown_rag_subprocess

    _teardown_rag_subprocess()


def retrieval_api_base_url() -> str:
    """Return the Week 2 retrieval endpoint configured for this process."""
    _prepare_import_path()
    from src.helper.helper import get_settings as get_week2_settings

    return str(get_week2_settings().RETRIEVAL_API_BASE_URL).rstrip("/")
