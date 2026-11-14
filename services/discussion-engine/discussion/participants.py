"""Deterministic participant selection for a discussion run."""
from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import Any


class ParticipantSelectionError(ValueError):
    """Raised when the configured participant count cannot be satisfied."""


def select_participant_ids(
    personas: Mapping[str, Any] | Iterable[str],
    *,
    num_participants: int,
    random_seed: int,
    selected_participant_ids: Iterable[str] | None = None,
) -> list[str]:
    """Select either an explicit roster or a reproducible random subset.

    Persona ids are sorted before sampling so the result does not depend on
    YAML or dictionary insertion order. The seeded sample order is retained
    because it also provides a reproducible ring order for the graph.
    """
    available = sorted(personas.keys() if isinstance(personas, Mapping) else personas)
    if num_participants < 3:
        raise ParticipantSelectionError("A discussion requires at least three participants.")
    if num_participants > len(available):
        raise ParticipantSelectionError(
            f"Requested {num_participants} participants, but only "
            f"{len(available)} personas are available."
        )

    if selected_participant_ids is not None:
        selected = list(selected_participant_ids)
        if len(selected) != num_participants:
            raise ParticipantSelectionError(
                "The custom participant count does not match the selected roster."
            )
        if len(selected) != len(set(selected)):
            raise ParticipantSelectionError("Custom participant ids must be unique.")
        unknown = [persona_id for persona_id in selected if persona_id not in available]
        if unknown:
            raise ParticipantSelectionError(
                "Unknown participant ids: " + ", ".join(unknown)
            )
        return selected

    return random.Random(random_seed).sample(available, k=num_participants)
