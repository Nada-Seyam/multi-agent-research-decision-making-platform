from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable
from .config import get_settings


class GraphValidationError(ValueError):
    """Raised when an ``AgentGraph`` is structurally invalid.

    Covers both "impossible" structural problems (edges referencing an
    unknown agent — though ``add_edge`` already rejects those immediately)
    and semantic problems (the graph is not strongly connected). Nothing in
    this module ever auto-repairs a graph that fails validation; callers
    must fix the graph (add agents/edges) and re-validate.
    """


class AgentGraph:
    """A directed graph of agent ids with a strong-connectivity check.

    Example
    -------
    >>> g = AgentGraph()
    >>> for agent_id in ("agent_a", "agent_b", "agent_c"):
    ...     g.add_agent(agent_id)
    >>> g.add_edge("agent_a", "agent_b")
    >>> g.add_edge("agent_b", "agent_c")
    >>> g.add_edge("agent_c", "agent_a")
    >>> g.is_strongly_connected()
    True
    """

    def __init__(self) -> None:
        # Insertion-ordered set of agent ids.
        self._agents: dict[str, None] = {}
        # adjacency[src] = insertion-ordered set of dst ids.
        self._adjacency: dict[str, dict[str, None]] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_agent(self, agent_id: str) -> None:
        """Register an agent (node). Safe to call more than once for the
        same id — adding an already-known agent is a no-op, not an error.
        """
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError(f"agent_id must be a non-empty string, got {agent_id!r}")
        if agent_id not in self._agents:
            self._agents[agent_id] = None
            self._adjacency[agent_id] = {}

    def add_edge(self, source_id: str, target_id: str) -> None:
        """Add a directed edge ``source_id -> target_id``.

        Raises ``KeyError`` immediately if either endpoint has not been
        added via ``add_agent`` first. This is deliberate: an edge to/from
        an unknown agent is a caller bug, not something this graph will
        silently paper over.
        """
        self._require_known(source_id)
        self._require_known(target_id)
        self._adjacency[source_id][target_id] = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_agents(self) -> list[str]:
        """Return all agent ids, in insertion order."""
        return list(self._agents.keys())

    def get_neighbors(self, agent_id: str) -> list[str]:
        """Return the ids that ``agent_id`` has a direct outgoing edge to."""
        self._require_known(agent_id)
        return list(self._adjacency[agent_id].keys())

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """Whether a direct edge ``source_id -> target_id`` exists.

        Returns ``False`` for unknown agents rather than raising, since
        this is a read-only existence check, not a mutation.
        """
        return target_id in self._adjacency.get(source_id, {})

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._agents

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    def is_strongly_connected(self) -> bool:
        """Whether every agent can reach every other agent.

        A graph with 0 or 1 agents is trivially strongly connected.
        Otherwise: pick any node, and check that (a) a forward
        traversal from it reaches every other node, and (b) a traversal
        over the *reversed* edges from it also reaches every other node.
        That two-traversal check is equivalent to, and much cheaper than,
        checking reachability between every pair of nodes.
        """
        agents = self.get_agents()
        if len(agents) <= 1:
            return True

        start = agents[0]
        forward_reachable = self._reachable_from(start, reversed_edges=False)
        backward_reachable = self._reachable_from(start, reversed_edges=True)

        all_agents = set(agents)
        return forward_reachable == all_agents and backward_reachable == all_agents

    def _reachable_from(self, start: str, *, reversed_edges: bool) -> set[str]:
        visited = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            if reversed_edges:
                neighbors = (
                    src for src, targets in self._adjacency.items() if node in targets
                )
            else:
                neighbors = self._adjacency[node].keys()
            for nxt in neighbors:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        return visited

    def validate(self) -> bool:
        """Validate the graph. Returns ``True`` on success.

        Raises ``GraphValidationError`` (without modifying the graph) if:
          - there are no agents, or
          - the graph is not strongly connected.

        Edges referencing an unknown agent can't actually occur here since
        ``add_edge`` rejects them at insertion time, but the check is kept
        as a defensive invariant in case the internal state is ever
        constructed another way (e.g. a future subclass).
        """
        if not self._agents:
            raise GraphValidationError("Graph has no agents.")

        known = set(self._agents.keys())
        for source_id, targets in self._adjacency.items():
            if source_id not in known:
                raise GraphValidationError(
                    f"Edge source {source_id!r} is not a registered agent."
                )
            for target_id in targets:
                if target_id not in known:
                    raise GraphValidationError(
                        f"Edge target {target_id!r} is not a registered agent."
                    )

        if not self.is_strongly_connected():
            raise GraphValidationError(
                "Agent graph is not strongly connected: at least one agent "
                "cannot reach, or cannot be reached by, some other agent."
            )
        return True

    def _require_known(self, agent_id: str) -> None:
        if agent_id not in self._agents:
            available = ", ".join(sorted(self._agents)) or "<none>"
            raise KeyError(
                f"Unknown agent_id: {agent_id!r}. Available agents: {available}"
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the graph.

        Shape::

            {
                "agents": ["agent_a", "agent_b", "agent_c"],
                "edges": [["agent_a", "agent_b"], ["agent_b", "agent_c"], ...]
            }
        """
        edges = [
            [source_id, target_id]
            for source_id, targets in self._adjacency.items()
            for target_id in targets
        ]
        return {"agents": self.get_agents(), "edges": edges}

    @classmethod
    def from_dict(cls, data: dict) -> "AgentGraph":
        """Reconstruct a graph previously produced by ``to_dict()``.

        Agents are added first, then edges — so an edge referencing an id
        missing from ``data["agents"]`` raises ``KeyError``, exactly as it
        would via manual ``add_agent``/``add_edge`` calls.
        """
        graph = cls()
        for agent_id in data.get("agents", []):
            graph.add_agent(agent_id)
        for edge in data.get("edges", []):
            source_id, target_id = edge
            graph.add_edge(source_id, target_id)
        return graph


# ----------------------------------------------------------------------
# Week 2 integration helpers
# ----------------------------------------------------------------------
#
# These are thin, optional convenience functions for wiring the graph to
# Week 2's existing agents. They only ever read agent *ids* — never agent
# objects, LLM clients, retrieval, memory, etc. — so the ``AgentGraph``
# class above stays fully independent of Week 2 (and of any future
# orchestration/routing code).


def load_week2_agent_ids(week2_root: str | Path | None = None) -> list[str]:
    """Return the Week 2 persona ids (agent ids), sorted for determinism.

    The Week 2 import boundary is kept in ``week2_adapter`` so this module
    remains focused on graph construction and validation.
    """
    from .week2_adapter import load_week2_personas

    return sorted(load_week2_personas(week2_root).keys())


def build_ring_graph(agent_ids: Iterable[str]) -> AgentGraph:
    """Build a strongly connected graph by arranging agents in a ring.

    ``agent_a -> agent_b -> agent_c -> agent_a`` for 3 agents, generalizing
    to any number of agents (>= 1). A ring is the simplest topology that
    guarantees strong connectivity for an arbitrary set of agent ids
    without hardcoding how many there are, which matters here since Week 2
    personas are loaded from a config file and may grow over time.

    Raises ``ValueError`` if ``agent_ids`` is empty.
    """
    ids = list(agent_ids)
    if not ids:
        raise ValueError("Cannot build a graph with zero agents.")
    if len(set(ids)) != len(ids):
        raise ValueError("Cannot build a graph with duplicate agent ids.")

    graph = AgentGraph()
    for agent_id in ids:
        graph.add_agent(agent_id)

    for i, agent_id in enumerate(ids):
        next_id = ids[(i + 1) % len(ids)]
        if next_id != agent_id:  # avoid a pointless self-loop when len(ids) == 1
            graph.add_edge(agent_id, next_id)

    return graph


def build_connected_graph(
    agent_ids: Iterable[str],
    *,
    random_seed: int | None = None,
) -> AgentGraph:
    """Build a reproducible sparse graph with mutual neighbor visibility."""
    ids = list(agent_ids)
    if not ids:
        raise ValueError("Cannot build a graph with zero agents.")
    if len(set(ids)) != len(ids):
        raise ValueError("Cannot build a graph with duplicate agent ids.")

    shuffled = ids[:]
    if random_seed is None:
        configured = get_settings().with_selected_random_seed()
        assert configured.random_seed is not None
        seed = configured.random_seed
    else:
        seed = random_seed
    random.Random(seed).shuffle(shuffled)
    graph = AgentGraph()
    for agent_id in ids:
        graph.add_agent(agent_id)

    for source_id, target_id in zip(shuffled, shuffled[1:]):
        graph.add_edge(source_id, target_id)
        graph.add_edge(target_id, source_id)

    graph.validate()
    return graph


def build_week2_agent_graph(
    agent_ids: Iterable[str] | None = None,
    *,
    week2_root: str | Path | None = None,
    random_seed: int | None = None,
) -> AgentGraph:
    """Build a sparse, strongly connected graph for selected personas.

    When ``agent_ids`` is omitted, all persona ids are loaded from Week 2.
    The discussion runner passes its seed-selected participant ids,
    so inactive personas are not added to that run's graph.
    """
    ids = list(agent_ids) if agent_ids is not None else load_week2_agent_ids(week2_root)
    return build_connected_graph(ids, random_seed=random_seed)
