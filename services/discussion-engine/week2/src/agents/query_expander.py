from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.llm import _client as get_llm


class ExpandedQuery(BaseModel):
    """Structured output for the query expander."""

    reasoning: str = Field(
        description="Explain why the previous query missed and what the rewrite fixes."
    )
    expanded_query: str = Field(
        description=(
            "A rewritten version of the original query. Keep the core topic; "
            "add the missing technical entities, synonyms, or specificity the "
            "previous query lacked. Do NOT change the topic."
        ),
    )


QUERY_EXPANSION_SYSTEM_PROMPT = (
    "You are a query-rewriting specialist used inside a retrieval feedback loop.\n"
    "You will be given:\n"
    "  1. The original user query.\n"
    "  2. The chunks the retrieval step returned for that query.\n"
    "  3. The relevance grader's reasoning for why those chunks are not useful.\n\n"
    "Your job is to produce ONE rewritten query that, on the next retrieval "
    "attempt, will return chunks the grader accepts.\n\n"
    "CRITICAL RULES:\n"
    "1. Keep the original topic. Never invent a new question.\n"
    "2. Add concrete technical entities, synonyms, acronym expansions, or "
    "   specific sub-topics that were missing from the previous query.\n"
    "3. Remove conversational filler ('Should we...', 'We are evaluating...', etc.).\n"
    "4. If the grader said the chunks were about the wrong technology, narrow "
    "   the rewrite to explicitly name the target technology from the query.\n"
    "5. If the grader said the chunks were too generic, sharpen the rewrite "
    "   toward a specific sub-concept (e.g. a paper, a benchmark, a method).\n"
    "6. Do not pad. The rewrite should be a single dense search-engine-style "
    "   query, not a paragraph."
)


def _format_chunks_for_expander(chunks: list[dict[str, Any]]) -> str:
    """Render a chunk list as a compact preview for the expander LLM."""
    if not chunks:
        return "(no chunks returned)"
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        text = (c.get("embedding_text") or c.get("text") or "").strip()
        if len(text) > 400:
            text = text[:400] + "..."
        source = c.get("source_name") or c.get("chunk_id")
        parts.append(f"[{i}] source={source}\n{text}")
    return "\n\n".join(parts)


def expand_query(
    query: str,
    chunks: list[dict[str, Any]],
    grading_reasoning: str,
) -> str:
    """Rewrite ``query`` to better match the evidence on the next retrieval.

    Returns the rewritten query string. On any LLM/parsing failure the
    original query is returned unchanged so the loop can make progress
    rather than stalling.
    """
    if not query or not query.strip():
        return query

    chunks_preview = _format_chunks_for_expander(chunks)
    user_msg = (
        f"# Original query\n{query.strip()}\n\n"
        f"# Chunks the previous retrieval returned\n{chunks_preview}\n\n"
        f"# Why the grader rejected those chunks\n"
        f"{(grading_reasoning or 'No reasoning provided.').strip()}\n\n"
        "Rewrite the query so the next retrieval returns chunks the grader accepts. "
        "You must output valid JSON matching the schema:\n"
        '{\n  "reasoning": "...",\n  "expanded_query": "..."\n}'
    )

    try:
        import json
        from src.helper.helper import clean_json

        llm = get_llm()
        raw = llm.invoke(
            [
                SystemMessage(content=QUERY_EXPANSION_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]
        )
        content = raw.content
        if isinstance(content, list):
            content = "".join((b.get("text", "") if isinstance(b, dict) else str(b)) for b in content)
        
        cleaned = clean_json(str(content))
        data = json.loads(cleaned)
        response = ExpandedQuery.model_validate(data)

        if response and getattr(response, "expanded_query", None):
            rewritten = response.expanded_query.strip()
            if rewritten:
                return rewritten
    except Exception as exc:
        print(f"[QueryExpander] Expansion failed: {exc}. Using previous query.")

    return query
