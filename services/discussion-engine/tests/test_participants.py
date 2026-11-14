from __future__ import annotations

import pytest

from discussion.participants import ParticipantSelectionError, select_participant_ids


PERSONAS = {
    "capability_maximizer": {},
    "cost_sustainability_analyst": {},
    "practical_architect": {},
    "product_user_advocate": {},
    "safety_governance_advocate": {},
    "strategic_consensus_builder": {},
}


def test_selection_is_reproducible_from_seed_only():
    first = select_participant_ids(
        PERSONAS,
        num_participants=4,
        random_seed=42,
    )
    second = select_participant_ids(
        reversed(PERSONAS),
        num_participants=4,
        random_seed=42,
    )

    assert first == second
    assert len(first) == 4
    assert set(first).issubset(PERSONAS)


def test_selection_preserves_seeded_order_for_graph_construction():
    selected = select_participant_ids(
        PERSONAS,
        num_participants=4,
        random_seed=42,
    )

    assert selected == select_participant_ids(
        PERSONAS,
        num_participants=4,
        random_seed=42,
    )


@pytest.mark.parametrize(
    ("num_participants", "message"),
    [
        (2, "at least three"),
        (7, "only 6 personas"),
    ],
)
def test_selection_rejects_invalid_participant_count(num_participants: int, message: str):
    with pytest.raises(ParticipantSelectionError, match=message):
        select_participant_ids(
            PERSONAS,
            num_participants=num_participants,
            random_seed=42,
        )


def test_custom_selection_preserves_the_requested_roster_order():
    requested = [
        "product_user_advocate",
        "practical_architect",
        "safety_governance_advocate",
    ]

    selected = select_participant_ids(
        PERSONAS,
        num_participants=3,
        random_seed=42,
        selected_participant_ids=requested,
    )

    assert selected == requested


@pytest.mark.parametrize(
    ("requested", "message"),
    [
        (
            ["practical_architect", "practical_architect", "product_user_advocate"],
            "must be unique",
        ),
        (
            ["practical_architect", "product_user_advocate", "missing_persona"],
            "Unknown participant ids: missing_persona",
        ),
    ],
)
def test_custom_selection_rejects_an_invalid_roster(requested: list[str], message: str):
    with pytest.raises(ParticipantSelectionError, match=message):
        select_participant_ids(
            PERSONAS,
            num_participants=3,
            random_seed=42,
            selected_participant_ids=requested,
        )
