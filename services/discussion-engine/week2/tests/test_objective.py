"""Unit tests for the objective loader.

All tests run offline (no LLM, no network). They use the bundled
``src/configs/objective.example.txt`` as a fixture.
"""
from __future__ import annotations

import sys
from pathlib import Path

_FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))

import pytest

from src.agents.objective import (
    Objective,
    load_objective,
    _split_sections,
    _parse_core_topics,
    _bullets,
    _paragraph,
)


EXAMPLE_CONTENT = """OBJECTIVE
Design a Transformer model architecture.

PRIMARY CONSTRAINTS
- Must run on limited compute.

CORE TOPICS TO DISCUSS

1. FEED-FORWARD / EXPERT ARCHITECTURE
- Should we use MoE?
- How to handle routing?

2. ATTENTION
- subpoint

3. DEPTH
- subpoint

4. WIDTH
- subpoint

5. NORMALIZATION
- subpoint

6. POSITIONAL ENCODINGS
- subpoint

7. OPTIMIZER
- subpoint

8. DATASET
- subpoint

EVIDENCE REQUIREMENTS
- Must have citations.

DECISION CRITERIA
1. Model quality / capability
2. Training compute
3. Training memory
4. Inference compute
5. Inference memory
6. Implementation complexity
7. Stability
8. Scalability
9. Evidence strength
10. Fit with the rest of the system

STRICT NOTES
- No speculation.

FINAL GOAL
Build the best model.
"""

@pytest.fixture
def example_path(tmp_path):
    p = tmp_path / "objective.example.txt"
    p.write_text(EXAMPLE_CONTENT, encoding="utf-8")
    return p


# ── Helpers ──────────────────────────────────────────────────────────


def _write_objective(tmp_path: Path, *, missing_sections: tuple[str, ...] = ()) -> Path:
    """Build a minimal valid objective file, optionally dropping sections."""
    sections = {
        "OBJECTIVE": "OBJECTIVE\nBuild a model.\n",
        "PRIMARY CONSTRAINTS": "PRIMARY CONSTRAINTS\n- Limited compute.\n",
        "CORE TOPICS TO DISCUSS": (
            "CORE TOPICS TO DISCUSS\n\n1. ATTENTION\n- Multi-head vs GQA.\n- Sliding window.\n\n2. DEPTH\n- How deep?\n"
        ),
        "EVIDENCE REQUIREMENTS": "EVIDENCE REQUIREMENTS\n- Always cite.\n",
        "DECISION CRITERIA": "DECISION CRITERIA\n1. Quality\n2. Compute\n",
        "STRICT NOTES": "STRICT NOTES\n- No speculation.\n",
        "FINAL GOAL": "FINAL GOAL\nA practical model.\n",
    }
    for sec in missing_sections:
        sections.pop(sec, None)
    body = "\n".join(sections[k] for k in sections)
    p = tmp_path / "objective.txt"
    p.write_text(body, encoding="utf-8")
    return p


# ── Tests: example file ──────────────────────────────────────────────


def test_example_file_loads(example_path):
    obj = load_objective(example_path)
    assert isinstance(obj, Objective)
    assert obj.goal
    assert "Transformer" in obj.goal
    assert obj.final_goal
    assert obj.constraints, "constraints should be non-empty"
    assert obj.decision_criteria, "decision_criteria should be non-empty"
    assert obj.evidence_requirements, "evidence_requirements should be non-empty"
    assert obj.strict_notes, "strict_notes should be non-empty"


def test_example_topic_count(example_path):
    obj = load_objective(example_path)
    assert len(obj.core_topics) == 8, f"expected 8 topics, got {len(obj.core_topics)}"


def test_example_topic_ids_are_slugs(example_path):
    obj = load_objective(example_path)
    first = obj.core_topics[0]
    assert first.id == "feed_forward_expert_architecture"
    assert "FEED-FORWARD" in first.title.upper()


def test_example_topic_subpoints_present(example_path):
    obj = load_objective(example_path)
    ff = obj.core_topics[0]
    assert any("MoE" in sp for sp in ff.subpoints)
    assert any("routing" in sp.lower() for sp in ff.subpoints)


def test_example_decision_criteria_preserve_order(example_path):
    obj = load_objective(example_path)
    # "Model quality / capability" is the first numbered item in the
    # DECISION CRITERIA section (a leading prose line may precede it).
    assert any(c.lower().startswith("model quality") for c in obj.decision_criteria)
    # The 10 ranking criteria must appear in the same order as in the file.
    expected_first_words = [
        "model quality", "training compute", "training memory",
        "inference compute", "inference memory", "implementation complexity",
        "stability", "scalability", "evidence strength", "fit with",
    ]
    lowered = [c.lower() for c in obj.decision_criteria]
    last_idx = -1
    for prefix in expected_first_words:
        idx = next((i for i, c in enumerate(lowered) if i > last_idx and c.startswith(prefix)), -1)
        assert idx > last_idx, f"criterion {prefix!r} not found in order (last seen at {last_idx})"
        last_idx = idx


# ── Tests: error paths ──────────────────────────────────────────────


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_objective(tmp_path / "nope.txt")


def test_empty_file_raises(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_objective(p)


def test_missing_final_goal_uses_neutral_default(tmp_path):
    p = _write_objective(tmp_path, missing_sections=("FINAL GOAL",))
    objective = load_objective(p)
    assert objective.final_goal == "Provide the best evidence-backed answer to the objective."


# ── Tests: helper coverage ───────────────────────────────────────────


def test_split_sections_seven_keys(example_path):
    raw = example_path.read_text(encoding="utf-8")
    sections = _split_sections(raw)
    assert set(sections.keys()) == {
        "OBJECTIVE",
        "PRIMARY CONSTRAINTS",
        "CORE TOPICS TO DISCUSS",
        "EVIDENCE REQUIREMENTS",
        "DECISION CRITERIA",
        "STRICT NOTES",
        "FINAL GOAL",
    }


def test_bullets_accepts_mixed_bullets_and_plain_text():
    # Bullets ("- " / "* "), numbered items ("1. "), and plain lines all pass through.
    items = _bullets(["- one", "* two", "1. three", "plain text", "- four"])
    assert items == ["one", "two", "three", "plain text", "four"]


def test_bullets_drops_empty_lines():
    items = _bullets(["- one", "", "  ", "- two"])
    assert items == ["one", "two"]


def test_paragraph_joins_lines():
    assert _paragraph(["line one", "line two", "", "line three"]) == "line one line two line three"


def test_parse_core_topics_two_topics():
    body = [
        "1. ATTENTION",
        "- Multi-head vs GQA.",
        "- Sliding window.",
        "",
        "2. DEPTH",
        "- How deep?",
    ]
    topics = _parse_core_topics(body)
    assert len(topics) == 2
    assert topics[0].title == "ATTENTION"
    assert topics[0].subpoints == ["Multi-head vs GQA.", "Sliding window."]
    assert topics[1].id == "depth"


def test_short_id_uses_goal():
    obj = Objective(
        goal="Build a Transformer from scratch.",
        final_goal="done",
    )
    sid = obj.short_id()
    assert sid.startswith("build_a_transformer_from_scratch")
