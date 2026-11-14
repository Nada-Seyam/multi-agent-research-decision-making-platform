"""Generate a cautious final narrative from last-round opinion records only."""

from __future__ import annotations

import re
from collections.abc import Iterable

from . import week2_adapter
from .models import OpinionResult

SYSTEM_PROMPT = """You synthesize the final state of a multi-agent discussion.
Use only the final-round opinion records supplied by the user. Do not use
outside knowledge, the original objective, the research dossier, previous
rounds, or a separate decision summary.

Separate three things carefully:
1. recommendations actually made by agents;
2. technical claims or assumptions asserted inside their opinions; and
3. unresolved conditions, evidence gaps, or risks.

Never present an agent's technical assertion as independently proven fact.
Use attributed language such as "the final opinions argue" or "the agents
identify." Do not use absolute phrases such as "only solution," "impossible,"
"ensures," or "guarantees." Distinguish strict agreement on the canonical
architecture identifier from broader semantic convergence. Preserve material
disagreement, low confidence, decision-critical uncertainty, and conditional
switch criteria. Return concise plain text. Do not use Markdown syntax: no
hash-prefixed headings, asterisks, backticks, or bullet markers.
Use short plain-text section labels followed by ordinary sentences."""

_AGREEMENT_OVERSTATEMENTS = (
    re.compile(r"\b(?:the\s+)?agents\s+agree\b", re.IGNORECASE),
    re.compile(r"\bunanim(?:ous|ously)\b", re.IGNORECASE),
    re.compile(r"\bconsensus\s+exists\b", re.IGNORECASE),
    re.compile(r"\bthere\s+is\s+(?:a\s+)?consensus\b", re.IGNORECASE),
)
_ABSOLUTE_OVERSTATEMENTS = (
    re.compile(r"\bonly\s+(?:viable\s+)?solution\b", re.IGNORECASE),
    re.compile(r"\bonly\s+architecture\b", re.IGNORECASE),
    re.compile(r"\bimpossible\b", re.IGNORECASE),
    re.compile(r"\bensures?\b", re.IGNORECASE),
    re.compile(r"\bguarantees?\b", re.IGNORECASE),
)


class ConclusionQualityWarning(ValueError):
    """A usable discussion completed, but its generated synthesis was unsafe.

    This is intentionally distinct from provider/runtime failures.  Callers can
    complete the discussion with ``fallback_conclusion`` while surfacing the
    quality issue to the user as a warning.
    """

    def __init__(
        self,
        reasons: list[str],
        fallback_conclusion: str,
    ) -> None:
        self.reasons = tuple(reasons)
        self.fallback_conclusion = fallback_conclusion
        super().__init__(
            "The conclusion model retained unsupported certainty after one rewrite: "
            + "; ".join(reasons)
        )


def _strict_consensus(opinions: list[OpinionResult]) -> bool:
    architectures = [opinion.architecture for opinion in opinions]
    return bool(architectures) and all(architectures) and len(set(architectures)) == 1


def _opinion_records(opinions: list[OpinionResult]) -> str:
    records: list[str] = []
    for opinion in opinions:
        records.append(
            "\n".join(
                [
                    f"AGENT: {opinion.persona_id}",
                    f"CANONICAL_ARCHITECTURE: {opinion.architecture or 'not recorded'}",
                    f"POSITION: {opinion.position or 'not recorded'}",
                    f"CONFIDENCE: {opinion.confidence:.3f}",
                    f"DECISION_CRITICAL: {'yes' if opinion.decision_critical else 'no'}",
                    (
                        "CRITICAL_UNCERTAINTY: "
                        f"{opinion.critical_uncertainty or 'none recorded'}"
                    ),
                    f"OPINION:\n{opinion.opinion}",
                ]
            )
        )
    return "\n\n---\n\n".join(records)


def _consensus_instruction(
    opinions: list[OpinionResult],
    strict_consensus: bool,
) -> str:
    architecture_map = ", ".join(
        f"{opinion.persona_id}={opinion.architecture or 'not recorded'}"
        for opinion in opinions
    )
    if strict_consensus:
        return (
            "The canonical architecture identifiers match across all supplied "
            f"opinions ({architecture_map}). You may describe strict agreement only "
            "if the narrative opinions do not materially contradict it."
        )
    return (
        "The canonical architecture identifiers do not establish strict consensus "
        f"({architecture_map}). Do not say that the agents agree, are unanimous, or "
        "have consensus. Describe shared ideas as convergence and explicitly retain "
        "the naming or architectural distinction."
    )


def _fallback_conclusion(opinions: list[OpinionResult]) -> str:
    """Build a conservative conclusion solely from validated structured fields."""
    positions = "; ".join(
        f"{opinion.persona_id}: {opinion.position or opinion.architecture or 'not recorded'}"
        for opinion in opinions
    )
    if _strict_consensus(opinions):
        recommendation = (
            "All usable final-round records use the same canonical candidate identifier, "
            f"{opinions[0].architecture}. This records alignment in the agent outputs, not "
            "independent validation of their technical claims."
        )
    else:
        recommendation = f"The final-round positions remain distinct: {positions}."

    uncertainties = tuple(
        dict.fromkeys(
            opinion.critical_uncertainty.strip()
            for opinion in opinions
            if opinion.critical_uncertainty and opinion.critical_uncertainty.strip()
        )
    )
    uncertainty_text = (
        "The final records identify these decision-critical uncertainties: "
        + "; ".join(uncertainties)
        if uncertainties
        else (
            "No structured decision-critical uncertainty was recorded. Review the final "
            "agent messages and switch conditions before acting on the recommendation."
        )
    )
    return "\n\n".join(
        (
            "Recommendations",
            recommendation,
            "Technical Claims and Assumptions",
            (
                "The generated narrative did not pass the conclusion certainty check. "
                "Technical claims remain attributed to the individual final-round agent "
                "messages and are not independently established by this fallback summary."
            ),
            "Unresolved Conditions, Evidence Gaps, and Risks",
            uncertainty_text,
        )
    )
def _overstatement_reasons(text: str, *, strict_consensus: bool) -> list[str]:
    reasons: list[str] = []
    if not strict_consensus and any(
        pattern.search(text) for pattern in _AGREEMENT_OVERSTATEMENTS
    ):
        reasons.append("claims strict agreement despite differing architecture records")
    if any(pattern.search(text) for pattern in _ABSOLUTE_OVERSTATEMENTS):
        reasons.append("states a technical claim with prohibited absolute language")
    return reasons


def _generate_checked(
    prompt: str,
    *,
    strict_consensus: bool,
) -> tuple[str, list[str]]:
    text = week2_adapter.generate_text(SYSTEM_PROMPT, prompt).strip()
    if not text:
        raise ValueError("The conclusion model returned an empty response.")
    return text, _overstatement_reasons(text, strict_consensus=strict_consensus)


def generate_conclusion(opinions: Iterable[OpinionResult]) -> str:
    """Synthesize final opinions and reject unsupported certainty."""
    usable = [
        opinion for opinion in opinions if opinion.opinion and opinion.opinion.strip()
    ]
    if not usable:
        raise ValueError("Cannot generate a conclusion without final-round opinions.")

    strict_consensus = _strict_consensus(usable)
    records = _opinion_records(usable)
    consensus_instruction = _consensus_instruction(usable, strict_consensus)
    prompt = (
        "Synthesize a final conclusion from the following final-round opinion "
        "records only.\n\n"
        f"CONSENSUS CONSTRAINT:\n{consensus_instruction}\n\n"
        f"FINAL-ROUND OPINION RECORDS:\n{records}"
    )
    conclusion, reasons = _generate_checked(
        prompt,
        strict_consensus=strict_consensus,
    )
    if not reasons:
        return conclusion

    repair_prompt = (
        "Rewrite the draft conclusion because it violates these constraints:\n"
        + "\n".join(f"- {reason}" for reason in reasons)
        + "\n\nThe draft is not evidence. Use only the final-round opinion records "
        "below as evidence. Preserve caveats and use attributed, conditional "
        "language.\n\n"
        f"CONSENSUS CONSTRAINT:\n{consensus_instruction}\n\n"
        f"DRAFT TO REWRITE:\n{conclusion}\n\n"
        f"FINAL-ROUND OPINION RECORDS:\n{records}"
    )
    repaired, remaining_reasons = _generate_checked(
        repair_prompt,
        strict_consensus=strict_consensus,
    )
    if remaining_reasons:
        raise ConclusionQualityWarning(
            remaining_reasons,
            _fallback_conclusion(usable),
        )
    return repaired
