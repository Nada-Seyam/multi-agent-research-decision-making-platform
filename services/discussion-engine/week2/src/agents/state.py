"""Agent state definitions.

This module defines the ``AgentState`` TypedDict that flows through
the LangGraph state machine for each persona. The objective-driven
architecture replaces the former single-query ``topic`` field with
``objective``, ``research_dossier``, and related fields.
"""
from __future__ import annotations

from typing import TypedDict, Any


class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes.

    Required fields (must be set by the caller):
      persona_id      — which persona is running
      objective       — the structured Objective for this cycle
      research_dossier — the consolidated dossier for this cycle

    All other fields are populated by graph nodes during execution.
    """
    # Core Keys
    persona_id: str
    objective: Any    # Objective — typed as Any to avoid import cycles in TypedDict
    research_dossier: Any  # ResearchDossier

    # Prompt Keys
    _system: str
    _user: str

    # Memory
    memory_context: str

    # Output Keys
    opinion: str
    error: str
    citation_warnings: list[int]
