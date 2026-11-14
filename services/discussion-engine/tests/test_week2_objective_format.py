from __future__ import annotations

from pathlib import Path

from discussion.week2_adapter import load_objective


def test_week2_legacy_objective_format_loads_without_llm(tmp_path: Path):
    objective_file = tmp_path / "objective.txt"
    objective_file.write_text(
        "objective -> Compare two model designs.\n\n"
        "final_goal -> Recommend one implementable design.\n\n"
        "constraints:\n"
        "- Train from scratch.\n\n"
        "topics_to_research:\n\n"
        "1. architecture\n"
        "- Compare dense and sparse designs.\n\n"
        "required_cross_source_research:\n"
        "- Use RAG first and web fallback when needed.\n\n"
        "decision_criteria:\n"
        "- Quality\n"
        "- Cost\n\n"
        "strict_notes:\n"
        "- Cite evidence.\n\n"
        "required_final_output:\n"
        "- Architecture\n",
        encoding="utf-8",
    )

    objective = load_objective(objective_file, "week2")

    assert objective.goal == "Compare two model designs."
    assert objective.final_goal == "Recommend one implementable design."
    assert objective.constraints == ["Train from scratch."]
    assert objective.core_topics[0].id == "architecture"
    assert objective.evidence_requirements == [
        "Use RAG first and web fallback when needed."
    ]
