"""Week 3 multi-agent discussion components."""

from .graph import (
    AgentGraph,
    GraphValidationError,
    build_connected_graph,
    build_ring_graph,
)

__all__ = [
    "AgentGraph",
    "GraphValidationError",
    "build_connected_graph",
    "build_ring_graph",
]
