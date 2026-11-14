"""LangGraph state graph definition — per-persona opinion pipeline.

The graph is a simple two-node pipeline:
    build_prompt → call_model → END

Context resolution (retrieval + grading + query expansion + source escalation)
lives in ``resolve_context()`` and is invoked once per research query **before**
any persona runs through this graph. The ``grade_relevance`` node has been
removed — grading happens in the evidence-pool step, not per-persona.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents import llm, prompts
from src.agents.config import get_persona
from src.agents.state import AgentState


def _is_provider_limit(exc: Exception) -> bool:
    """Return whether the provider rejected the call because of a quota/limit."""
    text = str(exc).lower()
    return (
        "429" in text
        or "rate limit" in text
        or "usage limit" in text
        or "quota" in text
        or "too many requests" in text
    )


def build_prompt_node(state: AgentState) -> dict:
    """Render the system + user prompts from the objective and dossier."""
    persona = get_persona(state["persona_id"])
    return {
        "_system": prompts.build_system_prompt(persona),
        "_user": prompts.build_opinion_prompt(
            state["objective"],
            state["research_dossier"],
            memory_context=state.get("memory_context"),
        ),
    }


def call_model_node(state: AgentState) -> dict:
    """Invoke the LLM with the constructed persona prompts."""
    system_prompt = state.get("_system")
    user_prompt = state.get("_user")

    if not system_prompt or not user_prompt:
        persona = get_persona(state["persona_id"])
        system_prompt = prompts.build_system_prompt(persona)
        user_prompt = prompts.build_opinion_prompt(
            state["objective"],
            state["research_dossier"],
        )

    try:
        opinion = llm.chat(system_prompt, user_prompt)
        return {"opinion": opinion}
    except Exception as e:
        persona_id = state.get("persona_id", "unknown")
        if _is_provider_limit(e):
            print(
                f"[llm] Persona {persona_id} unavailable: provider limit reached",
                file=sys.stderr,
            )
            return {"opinion": None, "status": "provider_limit"}
        print(f"[llm] Persona {persona_id} failed to generate opinion: {e}", file=sys.stderr)
        return {"opinion": None, "status": "error", "error": str(e)}


def build_graph():
    """Compile the two-node state graph.

    Context resolution runs pre-graph in ``run.py`` / ``evidence_pool.py``.
    The graph itself only builds the prompt and calls the model.
    """
    g = StateGraph(AgentState)

    g.add_node("build_prompt", build_prompt_node)
    g.add_node("call_model", call_model_node)

    g.add_edge(START, "build_prompt")
    g.add_edge("build_prompt", "call_model")
    g.add_edge("call_model", END)

    return g.compile()


# Cache the compiled graph (singleton).
_graph = None
_graph_lock = Lock()


def _get_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                _graph = build_graph()
    return _graph


def run_for_persona(
    persona_id: str,
    objective: Any,
    dossier: Any,
    *,
    memory_context: str | None = None,
) -> dict:
    """Invoke the graph for one persona; return the opinion text and metadata.

    ``objective`` is the ``Objective`` instance for this cycle.
    ``dossier`` is the ``ResearchDossier`` instance (consolidated evidence).
    """
    state: AgentState = {
        "persona_id": persona_id,
        "objective": objective,
        "research_dossier": dossier,
    }
    if memory_context is not None:
        state["memory_context"] = memory_context

    result = _get_graph().invoke(state)

    # Log resource usage (JSON Lines — one object per persona per cycle)
    log_file = Path(__file__).resolve().parents[2] / "logs" / "query_resource_usage.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    log_entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "objective_id": objective.short_id(),
        "persona_id": persona_id,
    }
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Logging] Error writing log file: {e}")

    ret: dict[str, Any] = {
        "opinion": result.get("opinion"),
        "is_relevant": bool(result.get("opinion")),
        "status": result.get("status", "ok"),
    }
    if "error" in result:
        ret["error"] = result["error"]
    return ret
