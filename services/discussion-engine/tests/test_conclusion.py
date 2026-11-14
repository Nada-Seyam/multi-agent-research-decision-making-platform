from __future__ import annotations

import pytest

from discussion import conclusion
from discussion.models import OpinionResult


def test_conclusion_prompt_contains_only_supplied_final_opinions(monkeypatch):
    captured: dict[str, str] = {}

    def fake_generate_text(system_prompt: str, user_prompt: str) -> str:
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return "The final opinions support option A, with unresolved cost concerns."

    monkeypatch.setattr(conclusion.week2_adapter, "generate_text", fake_generate_text)
    opinions = (
        OpinionResult(
            persona_id="agent_a",
            round_number=4,
            opinion="I recommend option A because it is reliable.",
            architecture="option_a",
            position="Option A",
            confidence=0.7,
        ),
        OpinionResult(
            persona_id="agent_b",
            round_number=4,
            opinion="Option A is preferable, but its cost remains uncertain.",
            architecture="option_b",
            position="Option B with shared components",
            confidence=0.4,
            decision_critical=True,
            critical_uncertainty="Production cost is unknown.",
        ),
    )

    result = conclusion.generate_conclusion(opinions)

    assert result == (
        "The final opinions support option A, with unresolved cost concerns."
    )
    assert "agent_a" in captured["user"]
    assert "agent_b" in captured["user"]
    assert opinions[0].opinion in captured["user"]
    assert opinions[1].opinion in captured["user"]
    assert "agent_a=option_a" in captured["user"]
    assert "agent_b=option_b" in captured["user"]
    assert "do not establish strict consensus" in captured["user"]
    assert "Production cost is unknown." in captured["user"]
    assert "original objective" in captured["system"]


def test_conclusion_rewrites_unsupported_consensus_and_certainty(monkeypatch):
    responses = iter(
        [
            "The agents agree this is the only solution and it ensures reliability.",
            (
                "The final opinions substantially converge on graph-based memory, "
                "while retaining different architecture identifiers and unresolved "
                "reliability conditions."
            ),
        ]
    )
    prompts: list[str] = []

    def fake_generate_text(system_prompt: str, user_prompt: str) -> str:
        prompts.append(user_prompt)
        return next(responses)

    monkeypatch.setattr(conclusion.week2_adapter, "generate_text", fake_generate_text)
    opinions = (
        OpinionResult(
            persona_id="agent_a",
            round_number=4,
            opinion="I recommend tri-layer memory, subject to latency validation.",
            architecture="tri_layer_memory",
        ),
        OpinionResult(
            persona_id="agent_b",
            round_number=4,
            opinion="I retain graph-native memory with tri-layer deployment.",
            architecture="graph_native_memory",
        ),
    )

    result = conclusion.generate_conclusion(opinions)

    assert result.startswith("The final opinions substantially converge")
    assert len(prompts) == 2
    assert "draft is not evidence" in prompts[1]
    assert "differing architecture records" in prompts[1]


def test_conclusion_rejects_overstatement_after_rewrite(monkeypatch):
    monkeypatch.setattr(
        conclusion.week2_adapter,
        "generate_text",
        lambda system_prompt, user_prompt: "The only solution ensures reliability.",
    )

    with pytest.raises(
        conclusion.ConclusionQualityWarning,
        match="retained unsupported certainty",
    ) as caught:
        conclusion.generate_conclusion(
            [
                OpinionResult(
                    persona_id="agent_a",
                    round_number=4,
                    opinion="I recommend option A, subject to testing.",
                    architecture="option_a",
                )
            ]
        )

    assert caught.value.reasons == (
        "states a technical claim with prohibited absolute language",
    )
    assert "Recommendations" in caught.value.fallback_conclusion
    assert "option_a" in caught.value.fallback_conclusion


def test_conclusion_requires_at_least_one_usable_opinion(monkeypatch):
    called = False

    def fake_generate_text(system_prompt: str, user_prompt: str) -> str:
        nonlocal called
        called = True
        return "unused"

    monkeypatch.setattr(conclusion.week2_adapter, "generate_text", fake_generate_text)

    with pytest.raises(ValueError, match="without final-round opinions"):
        conclusion.generate_conclusion(
            [OpinionResult(persona_id="agent_a", round_number=4, opinion=None)]
        )

    assert called is False
