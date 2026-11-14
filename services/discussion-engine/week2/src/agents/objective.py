"""Objective loader — parse a plain-text objective file into a structured Pydantic model.

The objective file format uses seven uppercase section headers:
    OBJECTIVE
    PRIMARY CONSTRAINTS
    CORE TOPICS TO DISCUSS
    EVIDENCE REQUIREMENTS
    DECISION CRITERIA
    STRICT NOTES
    FINAL GOAL

Under "CORE TOPICS TO DISCUSS", numbered top-level entries (e.g. "1. FEED-FORWARD / EXPERT
ARCHITECTURE") become CoreTopic entries, and the "- " bullet lines under each become
subpoints.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from src.agents.llm import _client as get_llm
from src.helper.helper import clean_json


_SECTION_HEADERS = (
    "OBJECTIVE",
    "PRIMARY CONSTRAINTS",
    "CORE TOPICS TO DISCUSS",
    "EVIDENCE REQUIREMENTS",
    "DECISION CRITERIA",
    "STRICT NOTES",
    "FINAL GOAL",
)

_BULLET_PREFIXES = ("- ", "* ")
_NUMBERED_ITEM_RE = re.compile(r"^\d+\.\s+(.+)$")
_TOPIC_HEADER_RE = re.compile(r"^\s*\d+\.\s+(.+?)\s*$")
_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9]+")
_LEGACY_SECTION_RE = re.compile(r"^\s*([a-z][a-z0-9_]*)\s*(?::|->)\s*(.*)\s*$")
_LEGACY_SECTION_NAMES = {
    "objective",
    "final_goal",
    "constraints",
    "topics_to_research",
    "required_cross_source_research",
    "decision_criteria",
    "strict_notes",
    "required_final_output",
}
_MARKDOWN_HEADER_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
_BOLD_SECTION_RE = re.compile(r"^\s*\*\*(.+?)(?::)?\*\*\s*$")


# ── Models ───────────────────────────────────────────────────────────


class CoreTopic(BaseModel):
    """A single discussion topic with a derived id and its bullet subpoints."""

    id: str = Field(description="Slugified identifier, e.g. 'feed_forward_expert_architecture'.")
    title: str = Field(description="Original title text exactly as it appears in the file.")
    subpoints: list[str] = Field(default_factory=list, description="Bullet items under the topic.")


class Objective(BaseModel):
    """Structured representation of a research objective file."""

    goal: str = Field(description="Single-sentence goal from the OBJECTIVE section.")
    constraints: list[str] = Field(default_factory=list)
    core_topics: list[CoreTopic] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
    decision_criteria: list[str] = Field(default_factory=list)
    strict_notes: list[str] = Field(default_factory=list)
    final_goal: str = Field(description="Verbatim final-goal paragraph.")

    def short_id(self, max_len: int = 32) -> str:
        """Stable short identifier for logging / file names."""
        base = _slugify(self.goal)
        return (base or "objective")[:max_len]


# ── Small helpers ────────────────────────────────────────────────────


def _slugify(text: str) -> str:
    return _NON_ALNUM_RE.sub("_", text.strip().lower()).strip("_")


def _is_bullet(line: str) -> bool:
    return line.startswith(_BULLET_PREFIXES)


def _bullet_text(line: str) -> str:
    for prefix in _BULLET_PREFIXES:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return line.strip()


def _paragraph(lines: list[str]) -> str:
    """Join body lines into a single trimmed paragraph (used for OBJECTIVE / FINAL GOAL)."""
    return " ".join(line.strip() for line in lines if line.strip()).strip()


# ── Parsing ──────────────────────────────────────────────────────────


def _split_sections(raw: str) -> dict[str, list[str]]:
    """Split a raw objective file into a mapping of section -> list of body lines.

    Section headers are recognized as lines whose stripped text equals (case-sensitive)
    one of ``_SECTION_HEADERS``. The header line itself is dropped from the body.
    """
    body: dict[str, list[str]] = {header: [] for header in _SECTION_HEADERS}
    current: str | None = None

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped in _SECTION_HEADERS:
            current = stripped
            continue
        if current is not None:
            body[current].append(line)

    return body


def _bullets(lines: list[str]) -> list[str]:
    """Extract items from a list of body lines.

    Accepts three list forms:
      * "- item" / "* item" bullets
      * "1. item" / "12. item" numbered lists (common for DECISION CRITERIA)
      * Plain non-empty lines (e.g. when a section has no list markers)

    Returns the cleaned text of each item, in input order. Skips blank lines.
    """
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _is_bullet(stripped):
            out.append(_bullet_text(stripped))
            continue
        match = _NUMBERED_ITEM_RE.match(stripped)
        out.append(match.group(1).strip() if match else stripped)
    return out


def _parse_core_topics(lines: list[str]) -> list[CoreTopic]:
    """Parse numbered top-level entries and the bullets under them.

    Recognized as a new topic: a line that begins with `<digits>. ` at column 0.
    Everything between topic headers is treated as the previous topic's subpoints
    (bullet lines have their prefix stripped; plain text lines are kept as-is).
    """
    topics: list[CoreTopic] = []
    current: CoreTopic | None = None

    for line in lines:
        header_match = _TOPIC_HEADER_RE.match(line)
        if header_match:
            if current is not None:
                topics.append(current)
            current = CoreTopic(
                id=_slugify(header_match.group(1)) or "topic",
                title=header_match.group(1).strip(),
                subpoints=[],
            )
            continue

        if current is None:
            continue

        stripped = line.strip()
        if not stripped:
            continue
        current.subpoints.append(_bullet_text(stripped) if _is_bullet(stripped) else stripped)

    if current is not None:
        topics.append(current)

    return topics


def _parse_legacy_objective(raw: str) -> Objective:
    """Parse the lower-case ``key:``/``key ->`` objective format.

    Week 2's checked-in example uses this format, while older objectives use
    the uppercase section format handled by ``_split_sections``. Supporting
    both keeps the loader compatible with existing objective files and avoids
    an unnecessary LLM fallback for a structurally complete objective.
    """
    sections: dict[str, list[str]] = {name: [] for name in _LEGACY_SECTION_NAMES}
    current: str | None = None

    for line in raw.splitlines():
        match = _LEGACY_SECTION_RE.match(line)
        if match and match.group(1) in sections:
            current = match.group(1)
            inline_value = match.group(2).strip()
            if inline_value:
                sections[current].append(inline_value)
            continue
        if current is not None:
            sections[current].append(line)

    required = ("objective", "final_goal", "constraints", "topics_to_research")
    missing = [name for name in required if not any(line.strip() for line in sections[name])]
    if missing:
        raise ValueError(
            "Legacy objective is missing required sections: " + ", ".join(missing)
        )

    return Objective(
        goal=_paragraph(sections["objective"]),
        constraints=_bullets(sections["constraints"]),
        core_topics=_parse_core_topics(sections["topics_to_research"]),
        evidence_requirements=_bullets(sections["required_cross_source_research"]),
        decision_criteria=_bullets(sections["decision_criteria"]),
        strict_notes=_bullets(sections["strict_notes"]),
        final_goal=_paragraph(sections["final_goal"]),
    )


def _markdown_sections(raw: str) -> dict[str, list[str]]:
    # Free-form briefs commonly put the goal in opening paragraphs and use
    # bold labels rather than hash headings. Treat that preamble as the goal
    # and recognize both heading styles deterministically.
    sections: dict[str, list[str]] = {"goal": []}
    current: str | None = "goal"
    for line in raw.splitlines():
        match = _MARKDOWN_HEADER_RE.match(line) or _BOLD_SECTION_RE.match(line)
        if match:
            heading = re.sub(r"[^a-z0-9]+", "_", match.group(1).lower()).strip("_")
            current = {
                "objective": "goal",
                "primary_constraints": "constraints",
                "strict_notes": "strict_notes",
                "final_goal": "final_goal",
            }.get(heading, heading)
            sections.setdefault(current, [])
        elif line.strip().lower() == "the decision involves competing priorities:":
            current = "decision_criteria"
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(line)
    return sections


def _clean_markdown(value: str) -> str:
    return re.sub(r"[*_`]", "", value).strip()


def _parse_markdown_objective(raw: str) -> Objective:
    """Parse the repository's documented Markdown-style decision brief."""
    sections = _markdown_sections(raw)
    required = ("goal", "constraints")
    if any(not sections.get(name) for name in required):
        raise ValueError("Markdown objective is missing Objective or Primary Constraints")

    goal_lines = sections["goal"]
    goal_paragraph: list[str] = []
    for line in goal_lines:
        stripped = line.strip()
        if stripped.startswith(("* ", "- ")):
            break
        if stripped:
            goal_paragraph.append(_clean_markdown(stripped))

    constraint_values = [
        _clean_markdown(_bullet_text(line.strip()))
        for line in sections["constraints"]
        if _is_bullet(line.strip())
    ]
    requirements_lines = sections.get("decision_requirements", [])
    requirement_values = [
        _clean_markdown(_bullet_text(line.strip()))
        for line in requirements_lines
        if _is_bullet(line.strip())
    ]

    criteria = [
        _clean_markdown(_bullet_text(line.strip())).rstrip(",;")
        for line in sections.get("decision_criteria", [])
        if _is_bullet(line.strip())
    ]
    collecting_criteria = False
    if not criteria:
        for line in requirements_lines:
            stripped = line.strip()
            if "compare candidate strategies across" in stripped.lower():
                collecting_criteria = True
                continue
            if collecting_criteria and _is_bullet(stripped):
                # Nested bullets are criteria; the next unindented top-level bullet
                # ends this subsection.
                if len(line) - len(line.lstrip()) <= 1:
                    collecting_criteria = False
                else:
                    criteria.append(_clean_markdown(_bullet_text(stripped)).rstrip(",;"))

    topics = [
        CoreTopic(
            id="decision_requirements",
            title="Decision Requirements",
            subpoints=requirement_values,
        ),
    ]
    evidence_requirements = [
        value for value in requirement_values
        if any(term in value.lower() for term in ("evidence", "measured", "extrapolation", "assumption"))
    ] or ["Separate measured evidence, inference, and unsupported assumptions."]
    strict_notes = [
        _clean_markdown(_bullet_text(line.strip()))
        for line in sections.get("strict_notes", [])
        if _is_bullet(line.strip())
    ]
    final_goal = _clean_markdown(_paragraph(sections.get("final_goal", [])))
    return Objective(
        goal=" ".join(goal_paragraph).strip(),
        constraints=constraint_values,
        core_topics=topics,
        evidence_requirements=evidence_requirements,
        decision_criteria=criteria or requirement_values,
        strict_notes=strict_notes,
        final_goal=final_goal or "Provide the best evidence-backed answer to the objective.",
    )


# ── LLM fallback ─────────────────────────────────────────────────────


_LLM_SYSTEM_PROMPT = (
    "You are an expert technical researcher. Your task is to extract and structure the contents of a free-form "
    "objective file into a specific JSON schema representing a research objective.\n"
    "The schema requires:\n"
    "- goal (string): A single-sentence goal.\n"
    "- constraints (list of strings): Any constraints mentioned.\n"
    "- core_topics (list of objects with 'id' (slug), 'title' (string), and 'subpoints' (list of strings)).\n"
    "- evidence_requirements (list of strings): Required evidence or research methodology.\n"
    "- decision_criteria (list of strings): How to evaluate the alternatives.\n"
    "- strict_notes (list of strings): Strict requirements or rules.\n"
    "- final_goal (string): A summary of the final output required.\n\n"
    "If the raw text doesn't explicitly contain some of these fields, infer them intelligently based on the context.\n"
    "Output valid JSON ONLY matching the provided schema. No preamble, no markdown fences, no commentary."
)


def _extract_objective_via_llm(raw: str) -> Objective:
    """Fallback: use the LLM to structure a free-form text file into an Objective."""
    schema_str = json.dumps(Objective.model_json_schema(), indent=2)
    user_prompt = f"Extract the following text into the JSON schema.\n\nText:\n{raw}\n\nSchema:\n{schema_str}"

    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=_LLM_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        content = response.content
        if isinstance(content, list):
            content = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)

        data = json.loads(clean_json(str(content)))
        return Objective.model_validate(data)
    except Exception as exc:
        print(f"[objective] LLM extraction failed: {exc}", file=sys.stderr)
        raise ValueError("Objective file is missing required sections and LLM extraction failed.") from exc


# ── Public API ───────────────────────────────────────────────────────


def load_objective(path: str | Path) -> Objective:
    """Load and parse an objective file at *path*.

    Raises ``FileNotFoundError`` if the file is missing.
    If the file does not contain the required explicit uppercase sections,
    it falls back to LLM extraction to populate the fields.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Objective file not found: {p}")

    raw = p.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError(f"Objective file is empty: {p}")

    if re.search(r"^\s*objective\s*(?::|->)", raw, flags=re.MULTILINE):
        return _parse_legacy_objective(raw)

    markdown_sections = _markdown_sections(raw)
    if all(markdown_sections.get(name) for name in ("goal", "constraints")):
        return _parse_markdown_objective(raw)

    sections = _split_sections(raw)
    minimum = ("OBJECTIVE", "PRIMARY CONSTRAINTS", "STRICT NOTES")
    if any(not any(line.strip() for line in sections[header]) for header in minimum):
        return _extract_objective_via_llm(raw)

    return Objective(
        goal=_paragraph(sections["OBJECTIVE"]),
        constraints=_bullets(sections["PRIMARY CONSTRAINTS"]),
        core_topics=_parse_core_topics(sections["CORE TOPICS TO DISCUSS"]),
        evidence_requirements=_bullets(sections["EVIDENCE REQUIREMENTS"]),
        decision_criteria=_bullets(sections["DECISION CRITERIA"]),
        strict_notes=_bullets(sections["STRICT NOTES"]),
        final_goal=(
            _paragraph(sections["FINAL GOAL"])
            or "Provide the best evidence-backed answer to the objective."
        ),
    )
