"""Session memory — rolling summary, working buffer, and compaction engine.

This module manages cross-cycle memory for the objective-driven pipeline.
Key changes from the original single-query architecture:
  - ``record_execution`` (turn-based) is replaced by ``record_cycle`` (cycle-based).
  - ``build_memory_context`` now takes ``objective_id`` instead of a free-form
    ``current_query`` string.
  - ``update_memory_from_cycle`` is a high-level helper that writes the full
    cycle's results into memory (decisions_log, contested topics, open_questions).
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator
from src.helper.helper import clean_json

# Allow tests / callers to monkeypatch these to point at a temp dir.
FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]
MEMORY_DIR = FRAMEWORK_ROOT / "memory"
SESSIONS_DIR = MEMORY_DIR / "sessions"
SUMMARY_PATH = SESSIONS_DIR / "rolling_summary.json"
WORKING_BUFFER_PATH = SESSIONS_DIR / "working_buffer.json"


# Lazy import to avoid a hard import-time dependency on the rest of the
# framework. ``get_llm`` is replaced by tests via monkeypatch.
def _get_llm():  # pragma: no cover - tiny indirection
    from src.agents.llm import _client as get_llm
    return get_llm()


# ── Canonical empty skeleton ─────────────────────────────────────────

def _known_persona_ids() -> list[str]:
    """Return persona ids loaded from ``Personas.yaml``.

    Using the YAML-driven list (not a hard-coded list) so that adding a
    new persona to Personas.yaml automatically creates a bucket in the
    memory skeleton. Falls back to the two known personas if the YAML
    cannot be loaded at import time.
    """
    try:
        from src.agents.config import get_personas
        ids = list(get_personas().get("personas", {}).keys())
        if ids:
            return ids
    except Exception:
        pass
    return ["capability_maximizer", "practical_architect"]


def empty_memory_skeleton() -> dict[str, Any]:
    """Return the canonical empty rolling-summary structure."""
    return {
        "last_updated": "",
        "shared_facts": [],
        "decisions_log": [],
        "persona_notes": {pid: [] for pid in _known_persona_ids()},
        "open_questions": [],
    }


# ── Errors ────────────────────────────────────────────────────────────

class MemoryCorruptError(RuntimeError):
    """Raised when ``rolling_summary.json`` cannot be parsed as JSON."""


# ── Load / persist ───────────────────────────────────────────────────

def load_session() -> dict[str, Any]:
    """Load the rolling summary; return empty skeleton if missing.

    On a malformed file, the bytes are quarantined to
    ``rolling_summary.json.corrupt-<timestamp>`` and an empty skeleton
    is returned. The original file is never silently overwritten.
    """
    if not SUMMARY_PATH.exists():
        return empty_memory_skeleton()

    try:
        raw = SUMMARY_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise MemoryCorruptError(
            f"Could not read rolling summary at {SUMMARY_PATH}: {exc}"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _quarantine_corrupt_file(raw, exc)
        return empty_memory_skeleton()

    if not isinstance(data, dict):
        _quarantine_corrupt_file(raw, ValueError("top-level value is not a dict"))
        return empty_memory_skeleton()

    skeleton = empty_memory_skeleton()
    for key in ("last_updated", "shared_facts", "decisions_log", "open_questions"):
        if key in data and isinstance(data[key], type(skeleton[key])):
            skeleton[key] = data[key]

    notes = data.get("persona_notes")
    if isinstance(notes, dict):
        for pid, items in notes.items():
            if isinstance(items, list) and all(isinstance(x, str) for x in items):
                skeleton["persona_notes"].setdefault(pid, []).extend(items)

    return skeleton


def _quarantine_corrupt_file(raw: str, exc: Exception) -> Path | None:
    """Move the corrupt bytes aside so they're recoverable, return the path."""
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = SESSIONS_DIR / f"rolling_summary.json.corrupt-{stamp}"
    try:
        target.write_text(raw, encoding="utf-8")
    except OSError:
        return None

    sys.stderr.write(
        f"[memory] rolling_summary.json was corrupt ({exc!r}); "
        f"quarantined to {target.name} and starting with an empty skeleton.\n"
    )
    return target


def _atomic_write_json(path: Path, data: Any, *, tmp_prefix: str) -> None:
    """Write ``data`` as pretty JSON to ``path`` atomically (write, then rename)."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    fd, tmp_path = tempfile.mkstemp(dir=str(SESSIONS_DIR), suffix=".tmp", prefix=tmp_prefix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_rolling_summary(summary: dict[str, Any]) -> None:
    """Atomically write ``summary`` to ``rolling_summary.json``."""
    _atomic_write_json(SUMMARY_PATH, summary, tmp_prefix=".summary_")


def load_working_buffer() -> list[dict[str, Any]]:
    """Load the working buffer; return an empty list if missing or unreadable."""
    if not WORKING_BUFFER_PATH.exists():
        return []
    try:
        data = json.loads(WORKING_BUFFER_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_working_buffer(buffer: list[dict[str, Any]]) -> None:
    """Atomically write ``buffer`` to ``working_buffer.json``."""
    _atomic_write_json(WORKING_BUFFER_PATH, buffer, tmp_prefix=".wb_")


# ── Token estimation ─────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Cheap token estimation: ~4 chars per token. No new dependency."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ── Working buffer mechanics ─────────────────────────────────────────

def _turn_to_text(turn: dict[str, Any]) -> str:
    role = turn.get("role", "")
    content = turn.get("content", "")
    persona = turn.get("persona_id")
    if persona:
        return f"[{role}:{persona}] {content}"
    return f"[{role}] {content}"


def split_window(
    working_buffer: list[dict[str, Any]],
    window_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(old_runs_to_compact, recent_runs_to_keep_raw)``.

    Invariant: the returned ``recent`` list has at most ``window_size``
    entries, and concatenating ``old + recent`` reproduces the input order.
    """
    if window_size <= 0:
        return list(working_buffer), []
    if len(working_buffer) <= window_size:
        return [], list(working_buffer)
    cut = len(working_buffer) - window_size
    return list(working_buffer[:cut]), list(working_buffer[cut:])


def should_compact(
    working_buffer: list[dict[str, Any]],
    *,
    window_size: int,
    token_budget: int,
) -> bool:
    """True when the buffer exceeds the window OR the token budget."""
    if len(working_buffer) > max(0, int(window_size)):
        return True
    if token_budget and token_budget > 0:
        total = sum(_estimate_tokens(_turn_to_text(t)) for t in working_buffer)
        if total > int(token_budget):
            return True
    return False


# ── Compaction (LLM) ─────────────────────────────────────────────────

class MemoryCompactionResult(BaseModel):
    """Structured output the compactor LLM must return."""

    shared_facts: list[str] = Field(
        default_factory=list,
        description=(
            "Established facts supported by the supplied execution logs. "
            "No speculation. No invented facts. Merge duplicates."
        ),
    )
    decisions_log: list[str] = Field(
        default_factory=list,
        description=(
            "Architectural or project decisions explicitly established. "
            "Include the reason if it is available in the logs."
        ),
    )
    persona_notes: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Per-persona notes. Keys MUST be persona ids from the existing "
            "summary; do not invent new persona buckets. Only include a "
            "persona if the logs actually contain relevant info."
        ),
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Unresolved questions that may matter to future executions.",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of what was added / merged / dropped.",
    )

    @field_validator("persona_notes", mode="before")
    @classmethod
    def _coerce_persona_notes(cls, v: Any) -> Any:
        if isinstance(v, dict):
            for k, val in v.items():
                if isinstance(val, str):
                    v[k] = [val]
        return v


COMPACTION_SYSTEM_PROMPT = (
    "You are a memory compaction specialist for an engineering research pipeline.\n"
    "You will receive the existing rolling memory (possibly empty) and a set of "
    "older system executions that have fallen out of the working window.\n\n"
    "Your job: produce an UPDATED rolling memory that merges the new information "
    "into the existing one. You are not appending; you are curating.\n\n"
    "CRITICAL RULES — read carefully:\n"
    "  1. Extract ONLY information that is supported by the supplied "
    "     execution logs. Do not invent facts.\n"
    "  2. Do not infer user preferences unless they are clearly established.\n"
    "  3. Do not convert speculation or hedged statements into facts. "
    "     'I think X might work' is NOT a fact that 'X works'.\n"
    "  4. Do not create decisions that were never made.\n"
    "  5. Do not create persona notes that are unsupported by the execution logs.\n"
    "  6. Preserve uncertainty where uncertainty exists (e.g. 'possibly', "
    "     'considering', 'we should evaluate').\n"
    "  7. Merge duplicate / overlapping information instead of producing "
    "     duplicate entries. Prefer the more specific phrasing.\n"
    "  8. persona_notes keys MUST be a subset of the existing persona ids. "
    "     Do not introduce new personas.\n\n"
    "Memory is DERIVED STATE, not authoritative truth — when in doubt, leave it out."
)


def _format_existing_summary(summary: dict[str, Any]) -> str:
    """Render the existing rolling summary for the compactor prompt."""
    has_list_content = any(
        summary.get(k) for k in ("shared_facts", "decisions_log", "open_questions")
    )
    has_persona_content = any((summary.get("persona_notes") or {}).values())
    if not summary or (not has_list_content and not has_persona_content):
        return "(empty)"

    parts: list[str] = []
    for key, title in (
        ("shared_facts", "Established Facts"),
        ("decisions_log", "Decisions"),
        ("open_questions", "Open Questions"),
    ):
        items = summary.get(key) or []
        if items:
            parts.append(f"### {title}\n" + "\n".join(f"- {x}" for x in items))

    notes = summary.get("persona_notes") or {}
    if any(notes.values()):
        rendered = [
            f"  - {pid}:\n" + "\n".join(f"    * {x}" for x in items)
            for pid, items in notes.items()
            if items
        ]
        if rendered:
            parts.append("### Persona Notes\n" + "\n".join(rendered))

    return "\n\n".join(parts) if parts else "(empty)"


def _format_turns_for_compactor(turns: list[dict[str, Any]]) -> str:
    if not turns:
        return "(no runs to compact)"
    blocks: list[str] = []
    for i, run in enumerate(turns, 1):
        role = run.get("role", "?")
        content = (run.get("content") or "").strip()
        persona = run.get("persona_id")
        speaker = role + (f" ({persona})" if persona else "")
        ts = run.get("timestamp") or ""
        header = f"[{i}] {speaker}" + (f" — {ts}" if ts else "")
        blocks.append(f"{header}\n{content}")
    return "\n\n".join(blocks)


def compact(
    old_runs: list[dict[str, Any]],
    rolling_summary: dict[str, Any],
) -> dict[str, Any]:
    """Use the LLM to produce an updated rolling summary.

    Falls back to ``rolling_summary`` unchanged if the LLM call fails
    or returns unusable output. ``compact`` never invents data.
    """
    existing_text = _format_existing_summary(rolling_summary)
    turns_text = _format_turns_for_compactor(old_runs)
    known = ", ".join(sorted((rolling_summary.get("persona_notes") or {}).keys()))

    user_msg = (
        f"# Existing Rolling Memory\n{existing_text}\n\n"
        f"# Old Executions To Compact (most recent last)\n{turns_text}\n\n"
        f"# Known Persona IDs (use ONLY these as persona_notes keys)\n{known}\n\n"
        "Produce the UPDATED rolling memory as JSON. Merge, do not append. "
        "Drop unsupported speculation. Preserve the existing structure."
    )

    try:
        llm = _get_llm()
        messages = [
            SystemMessage(content=COMPACTION_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
        if hasattr(llm, "with_structured_output"):
            raw = llm.with_structured_output(MemoryCompactionResult).invoke(messages)
        else:
            raw = llm.invoke(messages)

        if isinstance(raw, MemoryCompactionResult):
            response = raw
        elif isinstance(raw, str):
            response = MemoryCompactionResult.model_validate_json(clean_json(raw))
        elif isinstance(raw, dict) or hasattr(raw, "shared_facts"):
            response = MemoryCompactionResult.model_validate(
                raw if isinstance(raw, dict) else vars(raw)
            )
        else:
            content = raw.content
            if isinstance(content, list):
                content = "".join(
                    (b.get("text", "") if isinstance(b, dict) else str(b))
                    for b in content
                )
            response = MemoryCompactionResult.model_validate_json(
                clean_json(str(content).strip())
            )
        return merge_compaction(rolling_summary, response)
    except Exception as exc:
        sys.stderr.write(f"[memory] Compaction LLM failed: {exc!r}\n")
        return rolling_summary


# ── Merge / dedup ────────────────────────────────────────────────────

_NORMALIZE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[\s\.,;:!\?]+$")


def _norm(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", text.strip().lower())


def _norm_phrase(text: str) -> str:
    """Normalize for *containment* comparisons: also drop trailing punctuation
    so that a phrase ending in '.' (e.g. "uses PostgreSQL.") is still
    recognized as contained in "uses PostgreSQL with pgvector."."""
    return _TRAILING_PUNCT_RE.sub("", _norm(text))


def _merge_string_list(existing: list[str], incoming: list[str]) -> list[str]:
    """Merge ``incoming`` into ``existing`` with dedup.

    Containment rule: a new item whose normalized, punctuation-trimmed
    phrase is contained in (or equals) an existing item's phrase is
    treated as the same fact; the longer / more specific surface form
    wins and the shorter is dropped. Otherwise append as a new entry.
    Preserves order: existing items first, then genuinely new ones.
    """
    out: list[str] = list(existing)
    for new in incoming:
        if not isinstance(new, str) or not new.strip():
            continue

        n_new, p_new = _norm(new), _norm_phrase(new)
        replaced = False
        for i, old in enumerate(out):
            n_old, p_old = _norm(old), _norm_phrase(old)
            if n_new == n_old:
                if len(new) > len(old):
                    out[i] = new
                replaced = True
                break
            if p_old and p_new and (p_new in p_old or p_old in p_new):
                if len(p_new) > len(p_old):
                    out[i] = new
                replaced = True
                break

        if not replaced:
            out.append(new)
    return out


def merge_compaction(
    existing: dict[str, Any],
    compaction: MemoryCompactionResult,
) -> dict[str, Any]:
    """Merge a compactor result into the existing summary.

    Only persona ids already present in ``existing['persona_notes']`` are
    accepted — the compactor is forbidden from introducing new personas.
    """
    merged: dict[str, Any] = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "shared_facts": _merge_string_list(
            list(existing.get("shared_facts") or []),
            list(compaction.shared_facts or []),
        ),
        "decisions_log": _merge_string_list(
            list(existing.get("decisions_log") or []),
            list(compaction.decisions_log or []),
        ),
        "persona_notes": {},
        "open_questions": _merge_string_list(
            list(existing.get("open_questions") or []),
            list(compaction.open_questions or []),
        ),
    }

    existing_notes = dict(existing.get("persona_notes") or {})
    allowed = set(existing_notes.keys())
    for pid, items in (compaction.persona_notes or {}).items():
        if pid not in allowed:
            continue
        merged["persona_notes"][pid] = _merge_string_list(
            list(existing_notes.get(pid) or []),
            list(items or []),
        )
    for pid in allowed:
        merged["persona_notes"].setdefault(pid, list(existing_notes.get(pid) or []))

    return merged


# ── Prompt-ready rendering ───────────────────────────────────────────

def _render_section(title: str, items: list[str]) -> str:
    if not items:
        return f"### {title}\n(none)"
    return f"### {title}\n" + "\n".join(f"- {x}" for x in items)


def build_memory_context(
    rolling_summary: dict[str, Any],
    working_buffer: list[dict[str, Any]],
    objective_id: str,
    persona_id: str,
) -> str:
    """Render the prompt-ready memory context for ``persona_id``.

    Uses ``objective_id`` (not a free-form query string) to filter
    the working buffer so only entries from prior cycles on the same
    objective are shown. This prevents leaking context from unrelated
    objectives into the current persona's reasoning.
    """
    notes_for_all = rolling_summary.get("persona_notes") or {}
    my_notes: list[str] = list(notes_for_all.get(persona_id) or [])

    shared_facts = list(rolling_summary.get("shared_facts") or [])
    decisions = list(rolling_summary.get("decisions_log") or [])
    open_qs = list(rolling_summary.get("open_questions") or [])

    history_lines: list[str] = []
    for run in working_buffer:
        role = run.get("role", "?")
        content = (run.get("content") or "").strip()
        if not content:
            continue

        # Filter by objective_id: only show entries from prior cycles on the same objective.
        run_oid = run.get("objective_id")
        if run_oid and run_oid != objective_id:
            continue

        # Skip agent entries — they belong to a persona that ran before the current one.
        if role == "agent" and "persona_id" in run:
            continue

        history_lines.append(f"[{role}]\n{content}")

    sections = [
        "## Prior Execution Context\n",
        "The following facts, decisions, and past research summaries were established in previous runs of this system. "
        "Use them to inform your recommendation.",
        _render_section("Established Facts", shared_facts),
        _render_section("Decisions Log", decisions),
        _render_section(f"Notes for {persona_id}", my_notes),
        _render_section("Open Questions", open_qs),
        _render_section("Recent Past Executions", history_lines),
    ]
    return "\n\n".join(sections)


# ── Misc helpers ─────────────────────────────────────────────────────

def make_turn(role: str, content: str, **meta: Any) -> dict[str, Any]:
    """Convenience constructor for a working-buffer entry."""
    turn: dict[str, Any] = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    turn.update({k: v for k, v in meta.items() if v is not None})
    return turn


# ── Cycle-based memory write-back ───────────────────────────────────

def _memory_settings() -> Any | None:
    """Return settings if memory is enabled, else ``None`` (caller should no-op).

    Lazy import mirrors ``_get_llm`` above: avoids a hard import-time
    dependency on the rest of the framework.
    """
    from src.helper.helper import get_settings
    settings = get_settings()
    return settings if settings.MEMORY_ENABLED else None


def record_cycle(
    objective_id: str,
    objective_summary: str,
    plan_text: str | None,
    persona_summaries: dict[str, str],
) -> None:
    """Record a complete research cycle and compact if necessary.

    Replaces the old turn-based ``record_execution``. Each call writes
    one cycle entry (role="cycle") plus one agent entry per persona.

    ``objective_id`` is used for filtering in ``build_memory_context``.
    ``objective_summary`` is the objective's goal string (for human-readable logs).
    ``plan_text`` is the JSON-serialized research plan (5 queries).
    ``persona_summaries`` maps persona_id -> short summary of the opinion.
    """
    settings = _memory_settings()
    if settings is None:
        return

    working_buffer = load_working_buffer()

    working_buffer.append(make_turn("cycle", objective_summary, objective_id=objective_id))
    if plan_text:
        working_buffer.append(make_turn("planner", plan_text, objective_id=objective_id))
    for pid, summary in persona_summaries.items():
        working_buffer.append(make_turn("agent", summary, persona_id=pid, objective_id=objective_id))

    if not should_compact(
        working_buffer,
        window_size=settings.MEMORY_WINDOW_SIZE,
        token_budget=settings.MEMORY_TOKEN_BUDGET,
    ):
        save_working_buffer(working_buffer)
        return

    rolling_summary = load_session()
    old_runs, keep_raw = split_window(working_buffer, window_size=settings.MEMORY_WINDOW_SIZE)
    new_summary = compact(old_runs, rolling_summary)
    write_rolling_summary(new_summary)
    save_working_buffer(keep_raw)


def update_memory_from_cycle(
    objective: Any,
    dossier: Any,
) -> None:
    """Write durable dossier-derived memory after a completed cycle.

    Called once per cycle after both persona opinions are generated.

    - Topics where personas disagree get a ``"Contested: ..."`` entry.
    - Items flagged INSUFFICIENT EVIDENCE or needing more evidence go to ``open_questions``.

    Persona responses are retained in the working buffer by ``record_cycle`` and
    are incorporated only during compaction; this function does not infer a
    consensus from raw model output.
    """
    settings = _memory_settings()
    if settings is None:
        return

    summary = load_session()

    # ── Extract contested topics from contradictions in dossier ────
    if dossier and hasattr(dossier, "contradictions"):
        for c in dossier.contradictions:
            desc = c.description if hasattr(c, "description") else str(c)
            summary["decisions_log"].append(f"Contested: {desc}")

    # ── Extract unresolved / insufficient evidence questions ────────
    if dossier and hasattr(dossier, "per_topic_summaries"):
        for tid, text in dossier.per_topic_summaries.items():
            if "INSUFFICIENT" in str(text).upper():
                summary["open_questions"].append(
                    f"Insufficient evidence for topic '{tid}' in objective '{objective.short_id()}'"
                )

    write_rolling_summary(summary)
