"""Research planner — turn an Objective into a bounded research plan.

The planner uses an LLM with structured JSON output. It is *planner-driven*:
the model itself chooses which of the objective's core topics to investigate this
cycle, given the current session memory. There is no hardcoded topic-to-query
mapping.

The output is a ``ResearchPlan`` whose ``queries`` list has the caller-requested
bounded size, validated on every return path. Queries are decision-level: the
planner groups related requirements instead of creating one search for every
objective bullet.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.objective import Objective


# ── Pydantic schemas ────────────────────────────────────────────────

class ResearchPlan(BaseModel):
    """Structured output from the research planner."""

    queries: list[str] = Field(
        description="The caller-requested number of targeted, search-engine-style retrieval queries."
    )
    rationale: str = Field(
        default="",
        description="Brief explanation of why these topics were chosen this cycle.",
    )


# ── System prompt ───────────────────────────────────────────────────

def _planner_system_prompt(query_limit: int) -> str:
    return (
    "You are a research-planning specialist inside a multi-cycle AI agent framework.\n"
    f"Given an objective and the current session memory, you produce EXACTLY {query_limit}\n"
    "research queries that the framework will execute in parallel against a retrieval\n"
    "service this cycle.\n\n"
    "CRITICAL RULES:\n"
    f"1. Output EXACTLY {query_limit} queries. No more, no fewer.\n"
    "2. The full query set must cover the overall decision goal. Queries are broad,\n"
    "   decision-level coverage units, not one-query-per-bullet, metric, claim, or\n"
    "   uncertainty. Group related requirements into comparative searches.\n"
    "3. Each query must target a distinct decision dimension. Do not produce\n"
    "   paraphrases, near-duplicates, or generic padding queries.\n"
    "4. Queries must be concrete, technical, search-engine-style — not vague or\n"
    "   conversational.\n"
    "5. Treat objective constraints as decision boundaries. A qualitative "
    "constraint such as 'GPU-friendly' or 'low latency' should guide the "
    "comparison; do not turn it into a search for an unspecified GPU count, "
    "latency SLA, or product requirement. Only search for exact numbers when "
    "the objective explicitly supplies or requests them.\n"
    "6. Prioritize in this order:\n"
    "   (a) Topics with the largest expected effect on the quality / compute trade-off.\n"
    "   (b) Topics that are NOT already resolved in the prior session memory\n"
    "       (shared_facts, decisions_log). Do not re-research what's already known.\n"
    "   (c) Topics with strong available evidence rather than pure speculation.\n"
    "7. Cover the candidate alternatives and the highest-impact decision dimensions\n"
    "   fairly. Reserve one query for counterevidence, limitations, or conditions\n"
    "   that would change the decision when the query budget allows it.\n"
    "8. Do not invent new topics that aren't in the objective's CORE TOPICS list.\n"
    "9. Keep every query anchored to the objective, candidate set, workload, and\n"
    "   constraints. Do not chase implementation details that arise during debate.\n"
    "10. This is the only retrieval plan for the discussion. It must produce a\n"
    "   sufficient shared dossier without relying on later follow-up searches.\n"
    "11. Order queries from most critical to least critical for the decision.\n\n"
    "12. Treat component techniques as supporting details unless the objective explicitly\n"
    "   makes them a separate architecture family to compare.\n\n"
    "Output valid JSON only. No preamble, no markdown fences, no commentary.\n"
    "Schema: {\"queries\": [\"...\"], \"rationale\": \"...\"}\n"
    )


# ── Helpers ─────────────────────────────────────────────────────────

def _memory_digest(memory_summary: dict[str, Any]) -> str:
    """Render the rolling-summary fields the planner should consider."""
    if not memory_summary:
        return "(no prior session memory)"

    parts: list[str] = []
    for key, title in (
        ("shared_facts", "Established Facts"),
        ("decisions_log", "Decisions Log"),
        ("open_questions", "Open Questions"),
    ):
        items = memory_summary.get(key) or []
        if items:
            parts.append(f"### {title}\n" + "\n".join(f"- {x}" for x in items))

    if not parts:
        return "(no prior session memory)"
    return "\n\n".join(parts)


def _objective_digest(objective: Objective) -> str:
    """Render the objective in a planner-friendly compact form."""
    topic_lines = []
    for i, t in enumerate(objective.core_topics, 1):
        subpoints = "; ".join(t.subpoints) if t.subpoints else "(no subpoints)"
        topic_lines.append(f"{i}. {t.title} (id={t.id}): {subpoints}")
    topics_block = "\n".join(topic_lines) if topic_lines else "(no core topics)"

    def bullets(values: list[str]) -> str:
        return "\n".join(f"- {value}" for value in values) if values else "(none)"

    return (
        f"# Objective\n{objective.goal.strip()}\n\n"
        f"# Constraints\n{bullets(objective.constraints)}\n\n"
        f"# Core Topics (use these as your topic universe)\n{topics_block}\n\n"
        f"# Evidence Requirements\n{bullets(objective.evidence_requirements)}\n\n"
        f"# Decision Criteria\n{bullets(objective.decision_criteria)}\n\n"
        f"# Strict Notes\n{bullets(objective.strict_notes)}\n\n"
        f"# Final Goal\n{objective.final_goal.strip()}\n"
    )


def _deterministic_fallback(
    objective: Objective,
    memory_summary: dict[str, Any],
    query_limit: int = 5,
) -> ResearchPlan:
    """Build a deterministic, decision-level plan when the LLM is unavailable.

    The fallback uses unresolved topic comparisons first, then broad cross-cutting
    queries for constraints, criteria, evidence quality, and counterevidence. It
    never pads with near-identical copies of the overall goal.
    """
    known = set()
    for key in ("shared_facts", "decisions_log", "open_questions"):
        for item in (memory_summary or {}).get(key) or []:
            if isinstance(item, str):
                known.update(re.findall(r"[a-zA-Z]+", item.lower()))

    chosen: list[str] = []
    chosen_ids: list[str] = []
    skipped_ids: set[str] = set()
    for topic in objective.core_topics:
        topic_words = set(re.findall(r"[a-zA-Z]+", topic.title.lower()))
        topic_words.update(w for sp in topic.subpoints for w in re.findall(r"[a-zA-Z]+", sp.lower()))
        if topic_words & known:
            skipped_ids.add(topic.id)
            continue
        if len(chosen) >= query_limit:
            break
        chosen.append(_topic_query(topic, objective))
        chosen_ids.append(topic.id)

    # Pad if we still have room.
    for topic in objective.core_topics:
        if len(chosen) >= query_limit:
            break
        if topic.id in chosen_ids or topic.id in skipped_ids:
            continue
        chosen.append(_topic_query(topic, objective))
        chosen_ids.append(topic.id)

    cross_cutting = _cross_cutting_queries(objective)
    for query in cross_cutting:
        if len(chosen) >= query_limit:
            break
        chosen.append(query)

    # A very small or unusual objective may not provide enough material for all
    # slots. Use distinct objective-level questions rather than repeated padding.
    while len(chosen) < query_limit:
        slot = len(chosen) + 1
        chosen.append(
            f"{objective.goal.strip()[:220]}: integrated comparative evidence across "
            f"the open solution space, workload constraints, decision criteria, and counterevidence "
            f"(coverage synthesis {slot})"
        )

    return ResearchPlan(
        queries=chosen[:query_limit],
        rationale=(
            "Deterministic fallback plan derived from objective.core_topics; "
            "topics already covered in memory were skipped."
        ),
    )


def _topic_query(topic: Any, objective: Objective) -> str:
    """Render one goal-anchored query without exploding a topic into details."""
    details = "; ".join(topic.subpoints[:4]) if topic.subpoints else "the main trade-offs"
    return (
        f"{objective.goal.strip()[:220]} — {topic.title}: compare the candidate "
        f"alternatives using evidence that jointly covers {details}"
    )


def _cross_cutting_queries(objective: Objective) -> list[str]:
    """Return general decision-level query slots shared by many objectives."""
    goal = objective.goal.strip()
    constraints = "; ".join(objective.constraints[:4]) or "the stated constraints"
    criteria = "; ".join(objective.decision_criteria[:4]) or "the decision criteria"
    requirements = "; ".join(objective.evidence_requirements[:3]) or "the required evidence"
    return [
        f"{goal}: compare the candidate approaches under {constraints}, focusing on {criteria}",
        f"{goal}: gather transferable evidence for {requirements}, including source limitations and applicability",
        f"{goal}: identify counterevidence, failure modes, and conditions that would change the preferred approach",
        f"{goal}: compare the strongest alternatives on the most consequential trade-offs without assuming a winner",
    ]


# ── Public API ──────────────────────────────────────────────────────


def plan_research(
    objective: Objective,
    memory_summary: dict[str, Any] | None = None,
    query_limit: int = 5,
) -> ResearchPlan:
    """Plan exactly ``query_limit`` research queries for this cycle.

    On any LLM failure or schema validation error, falls back to a deterministic
    plan derived from the objective's core topics.
    """
    query_limit = max(1, int(query_limit))
    memory_summary = memory_summary or {}
    user_msg = (
        f"{_objective_digest(objective)}\n\n"
        f"# Session Memory (prior cycles)\n"
        f"{_memory_digest(memory_summary)}\n\n"
        f"Produce EXACTLY {query_limit} research queries for this cycle. Output JSON only."
    )

    from src.agents.llm import _client as get_llm

    try:
        llm = get_llm()
        raw = llm.invoke([
            SystemMessage(content=_planner_system_prompt(query_limit)),
            HumanMessage(content=user_msg),
        ])
        content = raw.content
        if isinstance(content, list):
            content = "".join(
                (b.get("text", "") if isinstance(b, dict) else str(b))
                for b in content
            )
        content = str(content).strip()
        # Strip accidental markdown fences the model may have added.
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        data = json.loads(content)
        plan = ResearchPlan.model_validate(data)
        if len(plan.queries) != query_limit:
            raise ValueError(
                f"planner returned {len(plan.queries)} queries, expected exactly {query_limit}"
            )
        if not all(q.strip() for q in plan.queries):
            raise ValueError("planner returned empty queries")
        normalized_queries = {" ".join(q.casefold().split()) for q in plan.queries}
        if len(normalized_queries) != len(plan.queries):
            raise ValueError("planner returned duplicate or near-duplicate queries")
        print(
            f"[planner] Generated {query_limit} research queries.",
            file=sys.stderr,
        )
        return plan
    except Exception as exc:
        print(
            f"[planner] LLM planning failed ({type(exc).__name__}: {exc}). "
            f"Falling back to deterministic {query_limit}-query plan.",
            file=sys.stderr,
        )
        return _deterministic_fallback(objective, memory_summary, query_limit=query_limit)
