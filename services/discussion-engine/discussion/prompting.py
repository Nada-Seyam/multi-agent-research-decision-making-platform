"""Build the per-persona prompt for one synchronous discussion round."""

from __future__ import annotations

from typing import Any

from .decision import criteria_registry, effective_criterion_weights
from .models import OpinionResult


def build_discussion_context(
    round_number: int,
    persona: dict[str, Any],
    previous: OpinionResult | None,
    neighbor_opinions: dict[str, OpinionResult],
    *,
    objective: Any | None = None,
    challenger_id: str | None = None,
    proposed_alternatives: tuple[str, ...] = (),
) -> str:
    priorities = persona.get("priorities", [])
    if isinstance(priorities, list):
        priorities = ", ".join(str(item) for item in priorities)
    perspective_weights = persona.get("perspective_weights") or ""
    anchor = persona.get("default_bias") or persona.get("stance") or ""
    criteria = criteria_registry(objective) if objective is not None else ()
    operational_weights = (
        effective_criterion_weights(persona, objective) if objective is not None else {}
    )
    lines = [
        f"# Discussion Round {round_number}",
        (
            "This is a synchronous discussion. Only the routed peer opinions below are "
            "visible to you. They include direct graph neighbors and may include one "
            "designated strongest challenger; do not infer or invent opinions from "
            "other personas."
        ),
        "## Persona decision anchor",
        f"Priorities: {priorities or 'use the persona priorities in the system prompt'}",
        (
            "Perspective weights: "
            f"{str(perspective_weights).strip()[:600] or 'use the shared decision frame'}"
        ),
        f"Default bias: {str(anchor).strip()[:600]}",
        "## Open solution space",
        (
            "No candidate list is supplied by the objective. Propose the primary solution "
            "that best follows from your persona priorities and the shared evidence."
            if not proposed_alternatives
            else "Previously proposed alternatives are listed below for comparison, but "
            "they are not an allowed-answer list. You may introduce a better solution."
        ),
        *(f"- {item}" for item in proposed_alternatives),
        "## Objective criteria and effective persona weights",
        *(
            f"- {criterion_id} (weight={operational_weights.get(criterion_id, 0):.4f}): {text}"
            for criterion_id, text in criteria
        ),
    ]
    if previous is not None:
        lines.extend(
            [
                "## Your previous position",
                f"Position: {previous.position or 'not established'}",
                f"Architecture: {previous.architecture or 'not established'}",
                f"Components: {previous.components or 'none specified'}",
                f"Considered alternatives: {previous.considered_alternatives or 'none recorded'}",
                f"Switch test: {previous.switch_test or 'none recorded'}",
                f"Strongest alternative: {previous.strongest_alternative or 'none recorded'}",
                f"Criteria scores: {previous.criteria_scores or 'none recorded'}",
                f"Decisive criterion: {previous.decisive_criterion or 'none recorded'}",
                f"Update decision: {previous.update_decision or 'none recorded'}",
                f"Matrix winner: {previous.ranked_candidate or 'none recorded'}",
                (
                    "Ranking margin: "
                    f"{previous.ranking_margin if previous.ranking_margin is not None else 'unknown'}"
                ),
                f"Confidence: {previous.confidence:.2f}",
                f"Opinion: {previous.opinion or 'no usable opinion'}",
            ]
        )
    if neighbor_opinions:
        lines.append("## Routed peer opinions")
        for neighbor_id, opinion in neighbor_opinions.items():
            role = (
                "designated strongest challenger"
                if neighbor_id == challenger_id
                else "visible peer"
            )
            lines.extend(
                [
                    f"### {neighbor_id} ({role})",
                    f"Position: {opinion.position or 'not established'}",
                    f"Architecture: {opinion.architecture or 'not established'}",
                    f"Components: {opinion.components or 'none specified'}",
                    f"Considered alternatives: {opinion.considered_alternatives or 'none recorded'}",
                    f"Switch test: {opinion.switch_test or 'none recorded'}",
                    f"Strongest alternative: {opinion.strongest_alternative or 'none recorded'}",
                    f"Criteria scores: {opinion.criteria_scores or 'none recorded'}",
                    f"Decisive criterion: {opinion.decisive_criterion or 'none recorded'}",
                    f"Update decision: {opinion.update_decision or 'none recorded'}",
                    f"Confidence: {opinion.confidence:.2f}",
                    f"Opinion: {opinion.opinion or 'no usable opinion'}",
                ]
            )
    else:
        lines.append(
            "## Routed peer opinions\nNo usable routed peer opinions are available yet."
        )

    phase_instruction = {
        1: (
            "Work independently before considering the group. Discover and propose one primary "
            "solution from the goal, constraints, and evidence; the objective intentionally gives "
            "you no candidate list. Compare your proposal with at least one serious alternative "
            "that you identify yourself, scoring both against every criterion. Apply the effective "
            "persona weights above; do not choose first and score afterward."
        ),
        2: (
            "Directly challenge the designated strongest challenger when one is present. Name the "
            "peer, their architecture, and their decisive claim; classify your response as ACCEPT, "
            "PARTIALLY ACCEPT, or REJECT. Cite the evidence that changes or preserves the relevant "
            "criterion score, then show the old and new score. Do not merely note the concern or "
            "call it a trade-off. Target the challenger's published decisive criterion or switch "
            "condition before restating your own priorities: try to falsify their decisive score, "
            "satisfy or contradict their switch test with existing evidence, or identify the exact "
            "missing evidence that prevents either result. Re-run your weighted ranking after the "
            "score change. Give the challenger's architecture a fair comparison instead of "
            "treating it as a straw man."
        ),
        3: (
            "Perform an explicit update test against your direct neighbors and the shared "
            "evidence. Address the strongest numerical score correction or switch-test challenge "
            "made against your position; do not answer a different, easier objection. Compare your "
            "old and new ranking of the decision criteria. Change your "
            "substantive position if a counterargument changes the highest-weight criterion or "
            "removes a key assumption; otherwise defend it with a falsifiable reason. Do not "
            "keep the old position only for consistency, and do not change merely to appear "
            "cooperative. Run a counterfactual switch test: name the strongest alternative, "
            "the concrete condition that would make it win, and whether the current objective "
            "already satisfies that condition. Score the current family and the alternative "
            "on the decisive criteria before choosing retain, switch, or unresolved."
        ),
        4: (
            "Finalize your position independently. Re-evaluate the switch condition from your "
            "previous round: mark it satisfied, contradicted, or unresolved. Switch families "
            "if it is satisfied; retain your current family if it is contradicted; if unresolved, "
            "lower confidence and make the recommendation explicitly conditional. Re-score "
            "the current family and the strongest alternative before finalizing. Preserve "
            "disagreement instead of claiming consensus."
        ),
    }.get(
        round_number, "Re-evaluate your position against the available direct evidence."
    )
    lines.extend(
        [
            "## Response protocol",
            phase_instruction,
            (
                "Use your persona priorities as soft starting weights, not fixed conclusions. "
                "Evaluate each neighbor claim independently: state its strongest version, "
                "separate evidence from inference and assumption, explain why it is wrong, "
                "incomplete, or stronger than your view, then defend your own recommendation "
                "with the best applicable evidence. Name one weakness or uncertainty in your "
                "own position. Update your conclusion when the exchange changes the ranking of "
                "a criterion or exposes a failed assumption; never agree merely because another "
                "persona sounds confident, and never disagree merely to preserve a persona identity."
            ),
            (
                "ARCHITECTURE must contain a short, stable name for one primary solution. It is "
                "open-ended and may name a solution not proposed by peers. COMPONENTS may list "
                "optional choices such as GQA, RMSNorm, SwiGLU, quantization, or batching, but "
                "components must not change the architecture vote. POSITION is a short human-readable "
                "repeat of ARCHITECTURE for compatibility. Do not hide a compromise behind a long "
                "feature bundle, and do not treat an optional serving or training technique as a "
                "separate architecture. A combined design is valid only when the evidence supports "
                "that combination; explain the combination in the opinion and keep both fields short."
            ),
            (
                "If your solution is the same as a previously proposed alternative, reuse that "
                "exact identifier instead of shortening it to a generic family name. For example, "
                "do not replace a specific identifier with only 'attention' or 'model'."
            ),
            (
                "Your prose recommendation, ARCHITECTURE, and POSITION must name the same solution. "
                "Candidate evaluations and their utility ranking are considerations, not a substitute "
                "for your recommendation; if you choose something other than the matrix winner, state "
                "the trade-off that justifies your choice."
            ),
            (
                "State important workload assumptions explicitly and make recommendations conditional "
                "when the objective does not specify context length, hardware, traffic, or latency SLA."
            ),
            "After your concise opinion, put these markers on separate lines:",
            (
                "Emit one single-line record for your selected solution and at least one serious "
                "alternative (also evaluate any peer proposal that could plausibly beat your current choice):"
            ),
            (
                "CANDIDATE_EVAL: <candidate_id> | scores=<criterion_id>:<0..5>,... | "
                "evidence=<citation numbers or NONE> | counter=<citation numbers or NONE> | "
                "feasibility=feasible|conditional|infeasible|unknown | "
                "assumptions=<semicolon separated or NONE> | gaps=<semicolon separated or NONE> | "
                "combination=direct|partial|indirect|none|not_applicable"
            ),
            "ARCHITECTURE: <one canonical architecture family>",
            "COMPONENTS: <optional comma-separated implementation choices, or NONE>",
            "CONSIDERED: <architecture alternatives you evaluated, or NONE>",
            (
                "SWITCH_TEST: current=<current solution> | alternative=<different solution> | "
                "condition=<measurable condition under which the alternative wins> | "
                "effect=alternative_wins | status=satisfied|contradicted|unresolved"
            ),
            (
                f"For this round, SWITCH_TEST current MUST be {previous.architecture}. If you "
                "switch, alternative MUST equal the new ARCHITECTURE; if you retain, alternative "
                "must be a different serious solution."
                if previous and previous.architecture
                else "In round one, SWITCH_TEST current MUST equal ARCHITECTURE and alternative "
                "must be a different serious solution that you discovered independently."
            ),
            (
                "The switch condition must describe evidence that favors the alternative, not "
                "evidence that makes it fail."
            ),
            "STRONGEST_ALTERNATIVE: <one architecture family you seriously evaluated>",
            "CRITERIA_SCORES: <current and alternative scores from 0 to 5, briefly>",
            "DECISIVE_CRITERION: <criterion driving retain or switch>",
            "UPDATE_DECISION: retain, narrow, lower_confidence, or switch",
            "UPDATE_REASON: <why the strongest neighbor claim did or did not change your view>",
            "POSITION: <short human-readable repeat of ARCHITECTURE>",
            "CONFIDENCE: <number from 0 to 1>",
            "CHANGE_OF_MIND: yes only if your substantive position changed, otherwise no",
            "CHANGE_REASON: <one short reason if yes; otherwise NONE>",
            "CHALLENGED_CLAIM: <the architecture-level claim tested, or NONE>",
            "DECISION_CRITICAL: yes only if the unresolved claim could change the winner, otherwise no",
            "CRITICAL_UNCERTAINTY: <the unresolved decisive claim, or NONE>",
            (
                "The dossier is fixed for the entire debate. Record missing evidence as a gap or "
                "CRITICAL_UNCERTAINTY, but do not request or attempt additional document retrieval."
            ),
        ]
    )
    return "\n".join(lines)
