"""Unit tests for the Week 3 agent communication graph.

All tests run offline (no LLM, no network, no I/O other than importing
Week 2's Personas.yaml for the integration test at the bottom).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))

import pytest

from discussion.graph import (
    AgentGraph,
    GraphValidationError,
    build_connected_graph,
    build_ring_graph,
    build_week2_agent_graph,
    load_week2_agent_ids,
)


# ── Helpers ──────────────────────────────────────────────────────────


def make_cycle(*agent_ids: str) -> AgentGraph:
    """agent_ids[0] -> agent_ids[1] -> ... -> agent_ids[-1] -> agent_ids[0]"""
    g = AgentGraph()
    for agent_id in agent_ids:
        g.add_agent(agent_id)
    for i, agent_id in enumerate(agent_ids):
        g.add_edge(agent_id, agent_ids[(i + 1) % len(agent_ids)])
    return g


# ── Adding agents ────────────────────────────────────────────────────


def test_add_agent_registers_node():
    g = AgentGraph()
    g.add_agent("agent_a")
    assert g.get_agents() == ["agent_a"]
    assert "agent_a" in g
    assert len(g) == 1


def test_add_agent_is_idempotent():
    g = AgentGraph()
    g.add_agent("agent_a")
    g.add_agent("agent_a")
    assert g.get_agents() == ["agent_a"]


def test_add_agent_rejects_empty_id():
    g = AgentGraph()
    with pytest.raises(ValueError):
        g.add_agent("")


# ── Adding directed edges ────────────────────────────────────────────


def test_add_edge_between_known_agents():
    g = AgentGraph()
    g.add_agent("agent_a")
    g.add_agent("agent_b")
    g.add_edge("agent_a", "agent_b")
    assert g.has_edge("agent_a", "agent_b")
    assert not g.has_edge("agent_b", "agent_a")  # directed, not symmetric


def test_add_edge_unknown_source_fails():
    g = AgentGraph()
    g.add_agent("agent_b")
    with pytest.raises(KeyError):
        g.add_edge("unknown_agent", "agent_b")


def test_add_edge_unknown_target_fails():
    g = AgentGraph()
    g.add_agent("agent_a")
    with pytest.raises(KeyError):
        g.add_edge("agent_a", "unknown_agent")


# ── Neighbors ────────────────────────────────────────────────────────


def test_get_neighbors_returns_outgoing_edges_only():
    g = make_cycle("agent_a", "agent_b", "agent_c")
    assert g.get_neighbors("agent_a") == ["agent_b"]
    assert g.get_neighbors("agent_b") == ["agent_c"]
    assert g.get_neighbors("agent_c") == ["agent_a"]


def test_get_neighbors_unknown_agent_fails():
    g = AgentGraph()
    g.add_agent("agent_a")
    with pytest.raises(KeyError):
        g.get_neighbors("unknown_agent")


# ── Checking edges ───────────────────────────────────────────────────


def test_has_edge_true_and_false_cases():
    g = make_cycle("agent_a", "agent_b", "agent_c")
    assert g.has_edge("agent_a", "agent_b") is True
    assert g.has_edge("agent_a", "agent_c") is False


def test_has_edge_unknown_agent_returns_false_not_raise():
    g = AgentGraph()
    g.add_agent("agent_a")
    assert g.has_edge("agent_a", "nope") is False
    assert g.has_edge("nope", "agent_a") is False


# ── Strong connectivity ──────────────────────────────────────────────


def test_strongly_connected_cycle_passes():
    g = make_cycle("agent_a", "agent_b", "agent_c", "agent_d")
    assert g.is_strongly_connected() is True
    assert g.validate() is True


def test_linear_chain_is_not_strongly_connected():
    # agent_a -> agent_b -> agent_c, no way back to agent_a
    g = AgentGraph()
    for agent_id in ("agent_a", "agent_b", "agent_c"):
        g.add_agent(agent_id)
    g.add_edge("agent_a", "agent_b")
    g.add_edge("agent_b", "agent_c")

    assert g.is_strongly_connected() is False
    with pytest.raises(GraphValidationError):
        g.validate()


def test_disconnected_component_is_not_strongly_connected():
    # Two separate cycles, no edges between them.
    g = AgentGraph()
    for agent_id in ("agent_a", "agent_b", "agent_c", "agent_d"):
        g.add_agent(agent_id)
    g.add_edge("agent_a", "agent_b")
    g.add_edge("agent_b", "agent_a")
    g.add_edge("agent_c", "agent_d")
    g.add_edge("agent_d", "agent_c")

    assert g.is_strongly_connected() is False
    with pytest.raises(GraphValidationError):
        g.validate()


def test_isolated_agent_with_no_edges_is_not_strongly_connected():
    g = AgentGraph()
    g.add_agent("agent_a")
    g.add_agent("agent_b")
    g.add_agent("agent_c")
    g.add_edge("agent_a", "agent_b")
    g.add_edge("agent_b", "agent_a")
    # agent_c has no edges at all.

    assert g.is_strongly_connected() is False
    with pytest.raises(GraphValidationError):
        g.validate()


def test_single_agent_graph_is_trivially_strongly_connected():
    g = AgentGraph()
    g.add_agent("agent_a")
    assert g.is_strongly_connected() is True
    assert g.validate() is True


def test_empty_graph_fails_validation():
    g = AgentGraph()
    assert g.is_strongly_connected() is True  # vacuously true
    with pytest.raises(GraphValidationError):
        g.validate()  # ...but validate() still requires >=1 agent


# ── Serialization ────────────────────────────────────────────────────


def test_to_dict_is_json_serializable_and_round_trips():
    original = make_cycle("agent_a", "agent_b", "agent_c")
    data = original.to_dict()

    # Must survive an actual JSON round trip, not just be dict-shaped.
    dumped = json.dumps(data)
    reloaded_data = json.loads(dumped)

    rebuilt = AgentGraph.from_dict(reloaded_data)
    assert rebuilt.get_agents() == original.get_agents()
    for agent_id in original.get_agents():
        assert sorted(rebuilt.get_neighbors(agent_id)) == sorted(
            original.get_neighbors(agent_id)
        )
    assert rebuilt.validate() is True


def test_from_dict_with_unknown_edge_target_fails():
    data = {
        "agents": ["agent_a", "agent_b"],
        "edges": [["agent_a", "agent_ghost"]],
    }
    with pytest.raises(KeyError):
        AgentGraph.from_dict(data)


# ── Week 2 integration ───────────────────────────────────────────────


def test_build_ring_graph_is_strongly_connected_for_arbitrary_ids():
    g = build_ring_graph(["p1", "p2", "p3", "p4", "p5"])
    assert g.validate() is True


def test_build_ring_graph_rejects_empty_input():
    with pytest.raises(ValueError):
        build_ring_graph([])


def test_build_connected_graph_is_sparse_reproducible_and_has_no_isolates():
    first = build_connected_graph(["a", "b", "c", "d"], random_seed=7)
    second = build_connected_graph(["a", "b", "c", "d"], random_seed=7)

    assert first.validate() is True
    assert first.to_dict() == second.to_dict()
    assert len(first.to_dict()["edges"]) == 6  # three undirected edges, both directions
    assert all(first.get_neighbors(agent_id) for agent_id in first.get_agents())


def test_load_week2_agent_ids_returns_known_personas():
    ids = load_week2_agent_ids()
    assert "capability_maximizer" in ids
    assert "practical_architect" in ids


def test_build_week2_agent_graph_is_valid_and_uses_real_persona_ids():
    g = build_week2_agent_graph()
    ids = set(load_week2_agent_ids())
    assert set(g.get_agents()) == ids
    assert g.validate() is True
