from discussion.candidate_identity import build_candidate_aliases
from discussion.decision import synthesize_decision
from discussion.models import DiscussionRound, OpinionResult
from discussion.opinion_parser import parse_opinion_response
from discussion.round_execution import canonicalize_round_opinions


def _opinion(persona: str, architecture: str) -> OpinionResult:
    return OpinionResult(
        persona_id=persona,
        round_number=1,
        opinion=f"I recommend {architecture}",
        position=architecture,
        architecture=architecture,
    )


def test_safe_qualifier_aliases_do_not_merge_materially_different_candidates():
    aliases = build_candidate_aliases(
        ["softmax_attention", "standard_softmax_attention", "linear_attention"]
    )

    assert aliases["softmax_attention"] == "standard_softmax_attention"
    assert aliases["standard_softmax_attention"] == "standard_softmax_attention"
    assert aliases["linear_attention"] == "linear_attention"


def test_aliases_merge_reordered_components_without_fuzzy_synonyms():
    candidates = [
        "hybrid_ssm_transformer_moe",
        "hybrid_transformer_mamba_moe",
        "hybrid_moe_ssm_transformer",
        "hybrid_moe_ssm_transformer",
    ]

    aliases = build_candidate_aliases(candidates)

    assert {
        aliases["hybrid_ssm_transformer_moe"],
        aliases["hybrid_moe_ssm_transformer"],
    } == {
        "hybrid_moe_ssm_transformer"
    }
    assert aliases["hybrid_transformer_mamba_moe"] == (
        "hybrid_transformer_mamba_moe"
    )


def test_aliases_keep_different_component_sets_separate():
    aliases = build_candidate_aliases(
        ["hybrid_moe_ssm_transformer", "hybrid_moe_transformer"]
    )

    assert aliases["hybrid_moe_ssm_transformer"] == "hybrid_moe_ssm_transformer"
    assert aliases["hybrid_moe_transformer"] == "hybrid_moe_transformer"


def test_aliases_merge_generic_semantic_wrappers_and_explicit_expansion():
    aliases = build_candidate_aliases(
        [
            "moe",
            "moe_architecture",
            "standard_moe_model",
            "mixture_of_experts_design",
        ],
        preferred=("moe",),
    )

    assert set(aliases.values()) == {"moe"}


def test_semantic_wrappers_do_not_remove_technical_components():
    aliases = build_candidate_aliases(
        ["moe", "dense_moe_architecture", "sparse_moe_model"]
    )

    assert aliases["moe"] == "moe"
    assert aliases["dense_moe_architecture"] == "dense_moe_architecture"
    assert aliases["sparse_moe_model"] == "sparse_moe_model"


def test_round_batch_canonicalizes_every_agents_equivalent_candidate():
    registry: dict[str, str] = {}
    opinions = [
        _opinion("strategy", "softmax_attention"),
        _opinion("capability", "standard_softmax_attention"),
        _opinion("safety", "standard_softmax_attention"),
        _opinion("product", "standard_softmax_attention"),
    ]

    normalized = canonicalize_round_opinions(opinions, registry)

    assert {item.architecture for item in normalized} == {
        "standard_softmax_attention"
    }
    assert registry["softmax_attention"] == "standard_softmax_attention"


def test_later_round_alias_is_resolved_before_retain_or_switch_logic():
    previous = _opinion("strategy", "standard_softmax_attention")
    current = parse_opinion_response(
        "strategy",
        2,
        {
            "opinion": (
                "CANDIDATE_EVAL: softmax_attention | scores=quality:5 | evidence=1 | "
                "counter=2 | feasibility=feasible | assumptions=NONE | gaps=NONE | "
                "combination=direct\n"
                "CANDIDATE_EVAL: mamba_ssm | scores=quality:3 | evidence=3 | "
                "counter=4 | feasibility=conditional | assumptions=NONE | gaps=NONE | "
                "combination=direct\n"
                "ARCHITECTURE: softmax_attention\n"
                "SWITCH_TEST: current=softmax_attention | alternative=mamba_ssm | "
                "condition=retrieval parity is demonstrated | effect=alternative_wins | "
                "status=contradicted\n"
                "UPDATE_DECISION: retain"
            )
        },
        previous=previous,
        persona_weights={"quality": 1.0},
        known_candidates=("standard_softmax_attention", "mamba_ssm"),
    )

    assert current.architecture == "standard_softmax_attention"
    assert current.changed_mind is False
    assert current.switch_assessment is not None
    assert current.switch_assessment.current_candidate == "standard_softmax_attention"
    assert current.switch_assessment.valid is True


def test_generic_marker_is_refined_by_explicit_recommendation():
    opinion = parse_opinion_response(
        "safety",
        1,
        {
            "opinion": (
                "I recommend **Standard Softmax Attention**.\n"
                "CANDIDATE_EVAL: Attention | scores=quality:5 | evidence=1 | "
                "counter=2 | feasibility=feasible | assumptions=NONE | gaps=NONE | "
                "combination=direct\n"
                "CANDIDATE_EVAL: SSM | scores=quality:3 | evidence=3 | "
                "counter=4 | feasibility=conditional | assumptions=NONE | gaps=NONE | "
                "combination=direct\n"
                "ARCHITECTURE: Attention\n"
                "SWITCH_TEST: current=Attention | alternative=SSM | "
                "condition=recall parity is demonstrated | effect=alternative_wins | "
                "status=unresolved\n"
                "UPDATE_DECISION: retain"
            )
        },
        persona_weights={"quality": 1.0},
    )

    assert opinion.architecture == "standard_softmax_attention"
    assert opinion.candidate_assessments[0].candidate_id == (
        "standard_softmax_attention"
    )
    assert opinion.switch_assessment is not None
    assert opinion.switch_assessment.current_candidate == (
        "standard_softmax_attention"
    )
    assert opinion.switch_assessment.valid is True


def test_generic_recommendation_alias_is_preserved_for_round_analytics():
    generic = parse_opinion_response(
        "safety",
        1,
        {
            "opinion": (
                "I recommend **Standard Softmax Attention**.\n"
                "ARCHITECTURE: Attention"
            )
        },
    )
    registry: dict[str, str] = {}

    normalized = canonicalize_round_opinions(
        [generic, _opinion("product", "softmax_attention")],
        registry,
    )

    canonical = {item.architecture for item in normalized}
    assert len(canonical) == 1
    assert registry["attention"] in canonical
    assert registry["softmax_attention"] in canonical


def test_markdown_protocol_labels_do_not_turn_intro_into_candidate():
    opinion = parse_opinion_response(
        "practical_architect",
        1,
        {
            "opinion": (
                "As a Practical Systems Architect, my primary objective is to "
                "maximize system value while minimizing operational risk.\n\n"
                "**Primary Recommendation: Hybrid Bi-Temporal Graph-Vector Architecture**\n"
                "**CANDIDATE_EVAL:** Hybrid-BiTemporal-GV | scores=quality:5 | "
                "evidence=1 | counter=2 | feasibility=feasible | assumptions=NONE | "
                "gaps=NONE | combination=direct\n"
                "**CANDIDATE_EVAL:** Pure-Vector-Store | scores=quality:2 | "
                "evidence=3 | counter=4 | feasibility=infeasible | assumptions=NONE | "
                "gaps=NONE | combination=direct\n"
                "**ARCHITECTURE:** Hybrid Bi-Temporal Graph-Vector\n"
                "**POSITION:** Hybrid Bi-Temporal Graph-Vector\n"
                "**COMPONENTS:** Vector database, knowledge graph"
            )
        },
        persona_weights={"quality": 1.0},
    )

    assert opinion.architecture == "hybrid_bi_temporal_graph_vector"
    assert opinion.position == "Hybrid Bi-Temporal Graph-Vector"
    assert opinion.components == "Vector database, knowledge graph"
    assert len(opinion.candidate_assessments) == 2
    assert not opinion.architecture.startswith("as_a_practical_systems_architect")


def test_unstructured_role_introduction_is_never_used_as_architecture():
    opinion = parse_opinion_response(
        "practical_architect",
        1,
        {
            "opinion": (
                "As a Practical Systems Architect, my primary objective is to "
                "maximize system value while minimizing operational risk."
            )
        },
    )

    assert opinion.position is None
    assert opinion.architecture is None


def test_explicit_prose_recommendation_is_safe_fallback_without_marker():
    opinion = parse_opinion_response(
        "practical_architect",
        1,
        {"opinion": "After evaluating both options, I recommend Hybrid Graph-Vector."},
    )

    assert opinion.position == "hybrid_graph_vector"
    assert opinion.architecture == "hybrid_graph_vector"


def test_final_decision_uses_same_alias_resolution_as_rounds():
    opinions = (
        _opinion("strategy", "softmax_attention"),
        _opinion("capability", "standard_softmax_attention"),
        _opinion("safety", "standard_softmax_attention"),
        _opinion("product", "standard_softmax_attention"),
    )

    summary = synthesize_decision((DiscussionRound(3, opinions, ()),))

    assert summary.preferred_candidate == "standard_softmax_attention"
    assert summary.consensus_status == "consensus"
    assert summary.support_personas == (
        "strategy",
        "capability",
        "safety",
        "product",
    )
    assert summary.dissenting_positions == {}
    assert summary.strongest_alternative is None
