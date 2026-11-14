from types import SimpleNamespace

from src.agents.objective import Objective
from src.agents.prompts import build_opinion_prompt


def _objective(*, criteria=None, notes=None):
    return Objective(
        goal="Choose an architecture.",
        final_goal="Recommend one architecture.",
        decision_criteria=criteria or [],
        strict_notes=notes or [],
    )


def test_decision_criteria_are_rendered_without_strict_notes():
    prompt = build_opinion_prompt(
        _objective(criteria=["Latency", "Reliability"]),
        SimpleNamespace(summary="[1] Evidence."),
    )

    assert "# Decision Criteria (FIXED RUBRIC)" in prompt
    assert "1. Latency" in prompt
    assert "2. Reliability" in prompt


def test_missing_criteria_uses_safe_fallback_and_renders_notes():
    prompt = build_opinion_prompt(
        _objective(notes=["Preserve dissent."]),
        SimpleNamespace(summary="[1] Evidence."),
    )

    assert "1. Quality and capability" in prompt
    assert "2. Latency and resource efficiency" in prompt
    assert "# Strict Notes\n\n- Preserve dissent." in prompt
