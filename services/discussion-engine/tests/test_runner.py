from __future__ import annotations

import json
import re
from threading import Barrier
from types import SimpleNamespace

from discussion import orchestrator as runner
from discussion.config import DiscussionSettings
from discussion.models import SharedResearchContext
from discussion.opinion_parser import architecture_key, parse_opinion_response

PERSONAS = {
    "capability_maximizer": {},
    "cost_sustainability_analyst": {},
    "practical_architect": {},
    "product_user_advocate": {},
    "safety_governance_advocate": {},
    "strategic_consensus_builder": {},
}


def make_objective():
    return SimpleNamespace(
        goal="Choose a practical AI architecture",
        short_id=lambda: "choose_a_practical_ai_architecture",
    )


def make_settings(**overrides):
    values = {
        "num_participants": 4,
        "num_rounds": 4,
        "random_seed": 42,
        "week2_root": "week2",
        "persist_artifacts": False,
    }
    values.update(overrides)
    return DiscussionSettings(**values)


def install_fakes(monkeypatch, research):
    monkeypatch.setattr(
        runner.week2_adapter,
        "load_week2_personas",
        lambda root: PERSONAS,
    )
    monkeypatch.setattr(
        runner,
        "build_shared_research",
        lambda objective, **kwargs: research,
    )
    monkeypatch.setattr(
        runner,
        "generate_conclusion",
        lambda opinions: "Final conclusion from final-round opinions.",
    )


def test_discussion_runs_four_synchronous_rounds(monkeypatch):
    research = SharedResearchContext(
        plan=SimpleNamespace(queries=["q"]),
        evidence_pool=[],
        dossier=SimpleNamespace(summary="dossier"),
    )
    calls: list[tuple[str, str | None, object]] = []
    install_fakes(monkeypatch, research)

    def fake_run(persona_id, objective, dossier, *, memory_context=None):
        calls.append((persona_id, memory_context, dossier))
        return {
            "opinion": (
                f"Recommendation from {persona_id}.\n"
                f"POSITION: position-{persona_id}\n"
                "CONFIDENCE: 0.6\n"
                "CHANGE_OF_MIND: no\n"
                "CHANGE_REASON: NONE"
            )
        }

    monkeypatch.setattr(runner.week2_adapter, "run_for_persona", fake_run)
    result = runner.run_discussion(make_objective(), settings=make_settings())

    assert len(result.participants) == 4
    assert len(result.rounds) == 4
    assert all(
        len(discussion_round.opinions) == 4 for discussion_round in result.rounds
    )
    assert len(calls) == 16
    assert [item.round_number for item in result.rounds[0].opinions] == [1] * 4
    assert result.graph["directed"] is True
    assert all(
        round_data["opinion"] == f"Recommendation from {persona}."
        for persona, opinion in result.to_dict()["opinions"].items()
        for round_data in opinion["rounds"]
    )
    assert all(
        "position" in round_data and "confidence" in round_data
        for persona in result.to_dict()["opinions"].values()
        for round_data in persona["rounds"]
    )
    assert json.loads(result.to_json())["graph"] == result.graph


def test_personas_within_each_round_run_in_parallel(monkeypatch):
    research = SharedResearchContext(plan=object(), evidence_pool=[], dossier=object())
    install_fakes(monkeypatch, research)
    round_barrier = Barrier(4)

    def fake_run(persona_id, objective, dossier, *, memory_context=None):
        round_barrier.wait(timeout=2)
        return {"opinion": f"Opinion {persona_id}\nPOSITION: {persona_id}"}

    monkeypatch.setattr(runner.week2_adapter, "run_for_persona", fake_run)
    result = runner.run_discussion(
        make_objective(),
        settings=make_settings(max_parallel_agents=4),
    )

    assert all(
        [opinion.persona_id for opinion in discussion_round.opinions]
        == list(result.participants)
        for discussion_round in result.rounds
    )
    assert all(
        opinion.status != "error"
        for discussion_round in result.rounds
        for opinion in discussion_round.opinions
    )


def test_direct_neighbors_plus_one_designated_challenger_are_visible(monkeypatch):
    research = SharedResearchContext(plan=object(), evidence_pool=[], dossier=object())
    contexts: dict[tuple[int, str], str] = {}
    install_fakes(monkeypatch, research)

    def fake_run(persona_id, objective, dossier, *, memory_context=None):
        round_number = int(
            re.search(r"Discussion Round (\d+)", memory_context or "1").group(1)
        )
        contexts[(round_number, persona_id)] = memory_context or ""
        return {"opinion": f"Opinion {persona_id}\nPOSITION: {persona_id}"}

    monkeypatch.setattr(runner.week2_adapter, "run_for_persona", fake_run)
    result = runner.run_discussion(make_objective(), settings=make_settings())

    neighbors = {
        persona_id: set(
            target for source, target in result.graph["edges"] if source == persona_id
        )
        for persona_id in result.participants
    }
    for persona_id in result.participants:
        context = contexts[(2, persona_id)]
        for neighbor_id in neighbors[persona_id]:
            assert f"### {neighbor_id}" in context
        visible_non_neighbors = {
            non_neighbor
            for non_neighbor in set(result.participants)
            - neighbors[persona_id]
            - {persona_id}
            if f"### {non_neighbor}" in context
        }
        assert len(visible_non_neighbors) <= 1
        if visible_non_neighbors:
            assert "designated strongest challenger" in context

    for discussion_round in result.rounds:
        for message in discussion_round.messages:
            assert set(message.recipients) == neighbors[message.sender]


def test_objective_does_not_define_or_restrict_solution_candidates(monkeypatch):
    research = SharedResearchContext(plan=object(), evidence_pool=[], dossier=object())
    contexts: dict[tuple[int, str], str] = {}
    install_fakes(monkeypatch, research)

    def fake_run(persona_id, objective, dossier, *, memory_context=None):
        round_number = int(
            re.search(r"Discussion Round (\d+)", memory_context or "1").group(1)
        )
        contexts[(round_number, persona_id)] = memory_context or ""
        primary = f"proposal_{persona_id}"
        alternative = f"alternative_{persona_id}"
        return {
            "opinion": (
                f"CANDIDATE_EVAL: {primary} | scores=quality:4 | evidence=1 | "
                "counter=2 | feasibility=feasible | assumptions=NONE | gaps=NONE | combination=not_applicable\n"
                f"CANDIDATE_EVAL: {alternative} | scores=quality:3 | evidence=3 | "
                "counter=4 | feasibility=conditional | assumptions=NONE | gaps=NONE | combination=not_applicable\n"
                f"ARCHITECTURE: {primary}\n"
                f"STRONGEST_ALTERNATIVE: {alternative}\n"
                f"SWITCH_TEST: current={primary} | alternative={alternative} | "
                "condition=measured quality is higher | effect=alternative_wins | status=contradicted\n"
                "UPDATE_DECISION: retain"
            )
        }

    monkeypatch.setattr(runner.week2_adapter, "run_for_persona", fake_run)
    result = runner.run_discussion(make_objective(), settings=make_settings())

    first_context = contexts[(1, result.participants[0])]
    second_context = contexts[(2, result.participants[0])]
    assert "No candidate list is supplied by the objective" in first_context
    assert "Previously proposed alternatives" in second_context
    assert "not an allowed-answer list" in second_context
    assert result.configuration["solution_space"] == "open"
    assert "candidate_registry" not in result.configuration
    assert len(result.configuration["proposed_alternatives"]) >= len(
        result.participants
    )


def test_debate_uses_one_fixed_dossier_and_never_retrieves_details(monkeypatch):
    research = SharedResearchContext(
        plan=object(),
        evidence_pool=[],
        dossier=SimpleNamespace(summary="initial dossier"),
    )
    install_fakes(monkeypatch, research)
    dossiers: list[str] = []

    def fake_run(persona_id, objective, dossier, *, memory_context=None):
        dossiers.append(dossier.summary)
        return {
            "opinion": (
                f"Opinion {persona_id}\nPOSITION: dense_transformer\n"
                "DECISION_CRITICAL: yes\n"
                "CRITICAL_UNCERTAINTY: target-hardware p95 is unknown"
            )
        }

    monkeypatch.setattr(runner.week2_adapter, "run_for_persona", fake_run)
    monkeypatch.setattr(
        runner.week2_adapter,
        "build_evidence_pool",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("debate rounds must not retrieve evidence")
        ),
    )

    result = runner.run_discussion(make_objective(), settings=make_settings())

    assert dossiers == ["initial dossier"] * 16
    assert all(
        opinion.critical_uncertainty == "target-hardware p95 is unknown"
        for round_data in result.rounds
        for opinion in round_data.opinions
    )


def test_mind_change_count_is_based_on_stable_position(monkeypatch):
    research = SharedResearchContext(plan=object(), evidence_pool=[], dossier=object())
    install_fakes(monkeypatch, research)

    def fake_run(persona_id, objective, dossier, *, memory_context=None):
        round_number = int(
            re.search(r"Discussion Round (\d+)", memory_context or "1").group(1)
        )
        position = (
            "hybrid_transformer_ssm"
            if round_number >= 3
            else "efficient_attention_transformer"
        )
        if round_number == 1:
            switch_test = "NONE"
        elif round_number == 2:
            switch_test = (
                "hybrid_transformer_ssm; stronger evidence appears; contradicted"
            )
        elif round_number == 3:
            switch_test = "hybrid_transformer_ssm; stronger evidence appears; satisfied"
        else:
            switch_test = "efficient_attention_transformer; stronger contrary evidence appears; contradicted"
        return {
            "opinion": (
                f"Opinion {persona_id}\nPOSITION: {position}\n"
                "CONFIDENCE: 0.5\n"
                f"UPDATE_DECISION: {'switch' if round_number == 3 else 'retain'}\n"
                f"SWITCH_TEST: {switch_test}\n"
                f"CHANGE_OF_MIND: {'yes' if round_number >= 3 else 'no'}\n"
                "CHANGE_REASON: stronger evidence"
            )
        }

    monkeypatch.setattr(runner.week2_adapter, "run_for_persona", fake_run)
    result = runner.run_discussion(make_objective(), settings=make_settings())

    for persona_id, opinion in result.to_dict()["opinions"].items():
        assert opinion["mind_change_count"] == 1
        assert opinion["rounds"][2]["changed_mind"] is True


def test_declared_position_change_is_preserved_for_analytics(monkeypatch):
    research = SharedResearchContext(plan=object(), evidence_pool=[], dossier=object())
    install_fakes(monkeypatch, research)

    def fake_run(persona_id, objective, dossier, *, memory_context=None):
        round_number = int(
            re.search(r"Discussion Round (\d+)", memory_context or "1").group(1)
        )
        position = "dense transformer" if round_number == 1 else "sparse moe"
        return {"opinion": f"POSITION: {position}"}

    monkeypatch.setattr(runner.week2_adapter, "run_for_persona", fake_run)
    result = runner.run_discussion(make_objective(), settings=make_settings())

    second = result.rounds[1].opinions[0]
    assert second.changed_mind is True
    assert second.architecture == "sparse_moe"
    assert second.status == "unresolved_switch_condition"


def test_components_do_not_count_as_architecture_changes():
    first = parse_opinion_response(
        "persona",
        1,
        {
            "opinion": (
                "Prefer a sparse architecture.\n"
                "ARCHITECTURE: sparse_moe_transformer\n"
                "COMPONENTS: GQA\n"
                "CONSIDERED: dense_transformer\n"
            )
        },
    )
    second = parse_opinion_response(
        "persona",
        2,
        {
            "opinion": (
                "Keep the same architecture but change normalization.\n"
                "ARCHITECTURE: sparse_moe_transformer\n"
                "COMPONENTS: RMSNorm, SwiGLU\n"
                "CONSIDERED: dense_transformer\n"
            )
        },
        previous=first,
    )

    assert second.changed_mind is False
    assert second.architecture == first.architecture


def test_open_solution_names_are_slugified_without_domain_aliases():
    assert architecture_key("Novel Recurrent Design") == ("novel_recurrent_design")
    assert architecture_key("efficientattentiontransformer") == (
        "efficientattentiontransformer"
    )


def test_open_proposal_can_switch_using_previous_architecture_as_current():
    first = parse_opinion_response(
        "persona",
        1,
        {
            "opinion": (
                "CANDIDATE_EVAL: sparse_moe | scores=quality:4 | evidence=1 | "
                "counter=2 | feasibility=feasible | assumptions=NONE | gaps=NONE | "
                "combination=direct\n"
                "CANDIDATE_EVAL: efficient_attention | scores=quality:4.1 | evidence=3 | "
                "counter=4 | feasibility=feasible | assumptions=NONE | gaps=NONE | "
                "combination=direct\n"
                "ARCHITECTURE: sparse_moe\n"
                "SWITCH_TEST: current=sparse_moe | alternative=efficient_attention | "
                "condition=measured reliability is materially higher | "
                "effect=alternative_wins | status=unresolved\n"
                "UPDATE_DECISION: retain"
            )
        },
        persona_weights={"quality": 1.0},
        utility_tolerance=0.25,
    )
    second = parse_opinion_response(
        "persona",
        2,
        {
            "opinion": (
                "CANDIDATE_EVAL: sparse_moe | scores=quality:3 | evidence=1 | "
                "counter=2 | feasibility=conditional | assumptions=NONE | gaps=NONE | "
                "combination=direct\n"
                "CANDIDATE_EVAL: efficient_attention | scores=quality:5 | evidence=3 | "
                "counter=4 | feasibility=feasible | assumptions=NONE | gaps=NONE | "
                "combination=direct\n"
                "ARCHITECTURE: efficient_attention\n"
                "SWITCH_TEST: current=sparse_moe | alternative=efficient_attention | "
                "condition=measured reliability is materially higher | "
                "effect=alternative_wins | status=satisfied\n"
                "UPDATE_DECISION: switch\n"
                "CHANGE_REASON: peer evidence changed the reliability assessment"
            )
        },
        previous=first,
        persona_weights={"quality": 1.0},
        utility_tolerance=0.25,
    )

    assert first.architecture == "sparse_moe"
    assert first.status == "ok"
    assert first.switch_assessment is not None and first.switch_assessment.valid
    assert second.architecture == "efficient_attention"
    assert second.changed_mind is True
    assert second.update_decision == "switch"
    assert second.switch_assessment is not None and second.switch_assessment.valid


def test_switch_test_is_preserved_as_separate_metadata():
    opinion = parse_opinion_response(
        "persona",
        3,
        {
            "opinion": (
                "Keep the current architecture conditionally.\n"
                "ARCHITECTURE: dense_transformer\n"
                "CONSIDERED: sparse_moe_transformer\n"
                "SWITCH_TEST: sparse_moe_transformer; predictable p95 fails; unresolved\n"
                "STRONGEST_ALTERNATIVE: sparse_moe_transformer\n"
                "CRITERIA_SCORES: current=4; alternative=3; memory current=2; alternative=5\n"
                "DECISIVE_CRITERION: predictable p95 latency\n"
                "UPDATE_DECISION: unresolved\n"
            )
        },
    )

    assert opinion.switch_test.startswith("sparse_moe_transformer;")
    assert opinion.strongest_alternative == "sparse_moe_transformer"
    assert opinion.criteria_scores.startswith("current=4")
    assert opinion.decisive_criterion == "predictable p95 latency"
    assert opinion.update_decision == "lower_confidence"


def test_switch_metadata_cannot_replace_the_declared_recommendation():
    first = parse_opinion_response(
        "persona",
        1,
        {
            "opinion": "Prefer efficient attention.\nARCHITECTURE: efficient_attention_transformer"
        },
    )
    second = parse_opinion_response(
        "persona",
        2,
        {
            "opinion": (
                "Hybrid is more feasible.\n"
                "ARCHITECTURE: hybrid_transformer_ssm\n"
                "SWITCH_TEST: hybrid_transformer_ssm; prove retrieval parity; unresolved\n"
                "UPDATE_DECISION: switch\n"
            )
        },
        previous=first,
    )

    assert second.architecture == "hybrid_transformer_ssm"
    assert second.position == "hybrid_transformer_ssm"
    assert second.changed_mind is True
    assert second.update_decision == "switch"
    assert second.status == "unresolved_switch_condition"


def test_architecture_switch_is_allowed_when_condition_is_satisfied():
    first = parse_opinion_response(
        "persona",
        1,
        {
            "opinion": "Prefer efficient attention.\nARCHITECTURE: efficient_attention_transformer"
        },
    )
    second = parse_opinion_response(
        "persona",
        2,
        {
            "opinion": (
                "Switch to hybrid.\n"
                "ARCHITECTURE: hybrid_transformer_ssm\n"
                "SWITCH_TEST: hybrid_transformer_ssm; retrieval parity; satisfied\n"
                "UPDATE_DECISION: switch\n"
                "CHANGE_REASON: comparable retrieval evidence\n"
            )
        },
        previous=first,
    )

    assert second.architecture == "hybrid_transformer_ssm"
    assert second.changed_mind is True
    assert second.update_decision == "switch"


def test_incomplete_candidate_matrix_is_flagged_and_caps_confidence():
    opinion = parse_opinion_response(
        "persona",
        1,
        {
            "opinion": (
                "CANDIDATE_EVAL: dense_transformer | scores=quality:5 | evidence=1 | "
                "counter=2 | feasibility=feasible | assumptions=NONE | gaps=NONE | "
                "combination=not_applicable\n"
                "CANDIDATE_EVAL: sparse_moe_transformer | scores=quality:4 | evidence=3 | "
                "counter=4 | feasibility=feasible | assumptions=NONE | gaps=NONE | "
                "combination=not_applicable\n"
                "ARCHITECTURE: dense_transformer\nCONFIDENCE: 0.9"
            )
        },
        persona_weights={"quality": 0.5, "latency": 0.5},
    )

    assert opinion.status == "incomplete_candidate_evaluation"
    assert opinion.confidence <= 0.55


def test_declared_recommendation_is_not_replaced_by_matrix_winner():
    opinion = parse_opinion_response(
        "persona",
        1,
        {
            "opinion": (
                "I recommend the sequential hybrid because feasibility matters.\n"
                "CANDIDATE_EVAL: sequential_hybrid | scores=quality:3 | evidence=1 | "
                "counter=2 | feasibility=feasible | assumptions=NONE | gaps=NONE | "
                "combination=direct\n"
                "CANDIDATE_EVAL: dense_transformer | scores=quality:5 | evidence=3 | "
                "counter=4 | feasibility=infeasible | assumptions=NONE | gaps=NONE | "
                "combination=not_applicable\n"
                "ARCHITECTURE: sequential_hybrid\n"
                "POSITION: sequential_hybrid\n"
                "SWITCH_TEST: current=sequential_hybrid | alternative=dense_transformer | "
                "condition=long context is removed | effect=alternative_wins | "
                "status=unresolved\n"
                "UPDATE_DECISION: retain"
            )
        },
        persona_weights={"quality": 1.0},
        utility_tolerance=0.25,
    )

    assert opinion.architecture == "sequential_hybrid"
    assert opinion.position == "sequential_hybrid"
    assert opinion.ranked_candidate == "dense_transformer"
    assert opinion.status == "recommendation_utility_conflict"


def test_matrix_winner_does_not_manufacture_a_missing_recommendation():
    opinion = parse_opinion_response(
        "persona",
        1,
        {
            "opinion": (
                "CANDIDATE_EVAL: option_a | scores=quality:3 | evidence=1 | "
                "counter=2 | feasibility=feasible | assumptions=NONE | gaps=NONE | "
                "combination=not_applicable\n"
                "CANDIDATE_EVAL: option_b | scores=quality:5 | evidence=3 | "
                "counter=4 | feasibility=feasible | assumptions=NONE | gaps=NONE | "
                "combination=not_applicable"
            )
        },
        persona_weights={"quality": 1.0},
    )

    assert opinion.architecture is None
    assert opinion.ranked_candidate == "option_b"
    assert opinion.status == "missing_recommendation"


def test_confidence_cannot_rise_during_fixed_dossier_debate():
    matrix = (
        "CANDIDATE_EVAL: open_design_a | scores=quality:5 | evidence=1 | counter=2 | "
        "feasibility=feasible | assumptions=NONE | gaps=NONE | combination=not_applicable\n"
        "CANDIDATE_EVAL: open_design_b | scores=quality:4 | evidence=3 | counter=4 | "
        "feasibility=feasible | assumptions=NONE | gaps=NONE | combination=not_applicable\n"
    )
    first = parse_opinion_response(
        "persona",
        1,
        {"opinion": matrix + "ARCHITECTURE: open_design_a\nCONFIDENCE: 0.6"},
        persona_weights={"quality": 1.0},
    )
    second = parse_opinion_response(
        "persona",
        2,
        {"opinion": matrix + "ARCHITECTURE: open_design_a\nCONFIDENCE: 0.9"},
        previous=first,
        persona_weights={"quality": 1.0},
    )

    assert second.confidence == first.confidence


def test_near_tie_ranking_change_is_recorded_without_forcing_architecture_switch():
    first = parse_opinion_response(
        "persona",
        1,
        {
            "opinion": (
                "CANDIDATE_EVAL: efficient_attention_transformer | scores=quality:4,cost:4 | "
                "evidence=1 | counter=2 | feasibility=feasible | assumptions=NONE | gaps=NONE | combination=not_applicable\n"
                "CANDIDATE_EVAL: hybrid_transformer_ssm | scores=quality:5,cost:4 | "
                "evidence=3 | counter=4 | feasibility=conditional | assumptions=scale risk | gaps=NONE | combination=direct\n"
                "ARCHITECTURE: hybrid_transformer_ssm\nSWITCH_TEST: NONE\nUPDATE_DECISION: retain"
            )
        },
        persona_weights={"quality": 0.5, "cost": 0.5},
        utility_tolerance=0.6,
    )
    second = parse_opinion_response(
        "persona",
        2,
        {
            "opinion": (
                "CANDIDATE_EVAL: efficient_attention_transformer | scores=quality:5,cost:4 | "
                "evidence=5 | counter=2 | feasibility=feasible | assumptions=NONE | gaps=NONE | combination=not_applicable\n"
                "CANDIDATE_EVAL: hybrid_transformer_ssm | scores=quality:4,cost:4 | "
                "evidence=3 | counter=6 | feasibility=conditional | assumptions=scale risk | gaps=NONE | combination=direct\n"
                "ARCHITECTURE: hybrid_transformer_ssm\n"
                "SWITCH_TEST: current=hybrid_transformer_ssm | alternative=efficient_attention_transformer | "
                "condition=measured target-fleet latency improves materially | "
                "effect=alternative_wins | status=contradicted\n"
                "UPDATE_DECISION: retain"
            )
        },
        previous=first,
        persona_weights={"quality": 0.5, "cost": 0.5},
        utility_tolerance=0.6,
    )

    assert second.architecture == "hybrid_transformer_ssm"
    assert second.changed_mind is False
    assert second.ranked_candidate == "efficient_attention_transformer"
    assert second.ranking_changed is True
    assert second.near_tie is True
    assert second.update_decision == "narrow"
    assert second.status == "near_tie_ranking_update"
    assert second.material_update is True


def test_agent_failure_is_recorded_without_stopping_other_agents(monkeypatch):
    research = SharedResearchContext(plan=object(), evidence_pool=[], dossier=object())
    install_fakes(monkeypatch, research)
    failed_persona: list[str] = []

    def fake_run(persona_id, objective, dossier, *, memory_context=None):
        if not failed_persona:
            failed_persona.append(persona_id)
            raise RuntimeError("model unavailable")
        return {"opinion": f"Opinion {persona_id}"}

    monkeypatch.setattr(runner.week2_adapter, "run_for_persona", fake_run)
    result = runner.run_discussion(
        make_objective(), settings=make_settings(random_seed=1)
    )

    opinions = [
        item for discussion_round in result.rounds for item in discussion_round.opinions
    ]
    failed = next(item for item in opinions if item.persona_id == failed_persona[0])
    assert failed.round_number == 1
    assert failed.opinion is None
    assert failed.is_relevant is False
    assert "model unavailable" in (failed.error or "")
    assert len(opinions) == 16


def test_provider_limit_is_an_expected_null_opinion(monkeypatch):
    research = SharedResearchContext(plan=object(), evidence_pool=[], dossier=object())
    install_fakes(monkeypatch, research)
    monkeypatch.setattr(
        runner.week2_adapter,
        "run_for_persona",
        lambda persona_id, objective, dossier, *, memory_context=None: {
            "opinion": None,
            "is_relevant": False,
            "status": "provider_limit",
        },
    )

    result = runner.run_discussion(make_objective(), settings=make_settings())
    opinion = result.rounds[0].opinions[0]

    assert opinion.opinion is None
    assert opinion.status == "provider_limit"
    assert opinion.error is None
    assert result.to_dict()["opinions"][opinion.persona_id]["rounds"][0]["status"] == (
        "provider_limit"
    )


def test_log_paths_are_exposed_without_embedding_heavy_data(monkeypatch):
    research = SharedResearchContext(plan=object(), evidence_pool=[], dossier=object())
    monkeypatch.setattr(
        runner.week2_adapter, "load_week2_personas", lambda root: PERSONAS
    )
    monkeypatch.setattr(
        runner, "build_shared_research", lambda objective, **kwargs: research
    )
    monkeypatch.setattr(
        runner,
        "generate_conclusion",
        lambda opinions: "Final conclusion from final-round opinions.",
    )
    monkeypatch.setattr(
        runner.week2_adapter,
        "run_for_persona",
        lambda persona_id, objective, dossier, *, memory_context=None: {
            "opinion": persona_id
        },
    )

    result = runner.run_discussion(make_objective(), settings=make_settings())
    concise = result.to_dict()
    assert set(concise) == {
        "schema_version",
        "discussion_id",
        "status",
        "progress",
        "preparation",
        "started_at",
        "updated_at",
        "completed_at",
        "failure",
        "warnings",
        "objective",
        "core_topics",
        "participants",
        "graph",
        "configuration",
        "rounds",
        "opinions",
        "decision_summary",
        "logs",
        "conclusion",
    }
    assert concise["conclusion"] == "Final conclusion from final-round opinions."
    assert "evidence_pool" not in concise
    assert set(result.logs) >= {
        "dossier",
        "rejected_evidence",
        "messages",
    }
