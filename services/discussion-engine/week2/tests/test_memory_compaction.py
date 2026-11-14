from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))


# ── Helpers ─────────────────────────────────────────────────────────

def _redirect_memory_paths(monkeypatch, tmp_path):
    """Point memory.py's path constants at a per-test temp dir."""
    from src.agents import memory as mem

    monkeypatch.setattr(mem, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(mem, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(mem, "SUMMARY_PATH", tmp_path / "sessions" / "rolling_summary.json")
    monkeypatch.setattr(mem, "WORKING_BUFFER_PATH", tmp_path / "sessions" / "working_buffer.json")
    return mem


def _install_compactor_mock(monkeypatch, mem_mod, responses):
    """Patch the LLM hook inside memory.py to return queued payloads."""
    queue = list(responses)

    class _FakeStructured:
        def __init__(self, payload):
            self._payload = payload
        def invoke(self, _messages):
            return self._payload

    class _FakeLLM:
        def with_structured_output(self, _schema):
            return _FakeStructured(queue.pop(0))

    monkeypatch.setattr(mem_mod, "_get_llm", lambda: _FakeLLM())
    # Also patch the name the module exports to test that path.
    monkeypatch.setattr(mem_mod, "get_llm", lambda: _FakeLLM(), raising=False)


# ── Tests ───────────────────────────────────────────────────────────

# Test 1 — Empty memory
def test_load_session_returns_canonical_skeleton_when_missing(monkeypatch, tmp_path):
    mem = _redirect_memory_paths(monkeypatch, tmp_path)
    summary = mem.load_session()
    # Canonical schema
    assert set(summary.keys()) >= {
        "last_updated", "shared_facts", "decisions_log", "persona_notes", "open_questions"
    }
    assert summary["shared_facts"] == []
    assert summary["decisions_log"] == []
    assert summary["open_questions"] == []
    # A bucket per persona in Personas.yaml
    from src.agents.config import get_personas
    expected = set(get_personas().get("personas", {}).keys())
    assert expected.issubset(set(summary["persona_notes"].keys()))
    for pid, items in summary["persona_notes"].items():
        assert items == []


# Test 2 — Window threshold
def test_should_compact_window_threshold():
    from src.agents.memory import should_compact, make_turn
    buf = [make_turn("user", f"q{i}") for i in range(4)]
    assert not should_compact(buf, window_size=4, token_budget=10_000)
    buf.append(make_turn("user", "q4"))
    assert should_compact(buf, window_size=4, token_budget=10_000)


# Test 3 — Token threshold
def test_should_compact_token_threshold():
    from src.agents.memory import should_compact, make_turn
    # One giant turn; under the turn count but over the token budget.
    big = "x" * (4 * 3000)  # ~3000 tokens with the 4-chars/token heuristic
    buf = [make_turn("user", big)]
    assert should_compact(buf, window_size=4, token_budget=2000)


# Test 4 — Persona isolation
def test_persona_isolation_in_memory_context(monkeypatch, tmp_path):
    mem = _redirect_memory_paths(monkeypatch, tmp_path)
    summary = mem.empty_memory_skeleton()
    summary["persona_notes"] = {
        "capability_maximizer": [
            "Always prefer maximum parameter count.",
            "Favor speculative decoding for throughput.",
        ],
        "practical_architect": [
            "Reject anything without a clear ops story.",
            "Demand rollback plans for every change.",
        ],
    }

    cap_ctx = mem.build_memory_context(
        summary, working_buffer=[], objective_id="Q", persona_id="capability_maximizer"
    )
    pa_ctx = mem.build_memory_context(
        summary, working_buffer=[], objective_id="Q", persona_id="practical_architect"
    )

    # Capability notes must NOT appear in practical_architect context.
    for note in summary["persona_notes"]["capability_maximizer"]:
        assert note not in pa_ctx, f"leaked capability note: {note!r}"
    # Practical notes must NOT appear in capability_maximizer context.
    for note in summary["persona_notes"]["practical_architect"]:
        assert note not in cap_ctx, f"leaked practical note: {note!r}"
    # And the OWN notes should be present.
    assert "Always prefer maximum parameter count." in cap_ctx
    assert "Reject anything without a clear ops story." in pa_ctx


# Test 5 — JSON round trip across multiple compactions
def test_json_round_trip_multiple_compaction_cycles(monkeypatch, tmp_path):
    mem = _redirect_memory_paths(monkeypatch, tmp_path)

    # Cycle 1: empty start.
    summary = mem.load_session()
    assert summary["shared_facts"] == []

    cycle1 = SimpleNamespace(
        shared_facts=["The project uses PostgreSQL."],
        decisions_log=[],
        persona_notes={"capability_maximizer": [], "practical_architect": []},
        open_questions=["Which quantization format?"],
        reasoning="first cycle",
    )
    _install_compactor_mock(monkeypatch, mem, [cycle1])

    old = [
        {"role": "user", "content": "We picked Postgres.", "timestamp": "t1"},
        {"role": "assistant", "content": "Postgres works.", "timestamp": "t2"},
    ]
    merged = mem.compact(old, summary)
    mem.write_rolling_summary(merged)

    # Reload
    reloaded = mem.load_session()
    assert "The project uses PostgreSQL." in reloaded["shared_facts"]
    assert "Which quantization format?" in reloaded["open_questions"]
    # last_updated updated
    assert reloaded["last_updated"]

    # Cycle 2: compaction extends facts.
    cycle2 = SimpleNamespace(
        shared_facts=["The project uses PostgreSQL with pgvector."],
        decisions_log=["Quantization format: INT4 AWQ."],
        persona_notes={
            "capability_maximizer": ["Wants max model size."],
            "practical_architect": ["Wants simple deployment."],
        },
        open_questions=[],
        reasoning="second cycle",
    )
    _install_compactor_mock(monkeypatch, mem, [cycle2])

    old2 = [
        {"role": "user", "content": "Also using pgvector.", "timestamp": "t3"},
    ]
    merged2 = mem.compact(old2, reloaded)
    mem.write_rolling_summary(merged2)

    final = mem.load_session()
    # Dedup should have collapsed to ONE postgres-related fact.
    pg_facts = [f for f in final["shared_facts"] if "PostgreSQL" in f or "Postgres" in f]
    assert len(pg_facts) == 1, f"expected 1 pg fact, got {pg_facts}"
    assert "INT4 AWQ" in final["decisions_log"][0]
    assert "Wants max model size." in final["persona_notes"]["capability_maximizer"]
    assert "Wants simple deployment." in final["persona_notes"]["practical_architect"]


# Test 6 — Deduplication
def test_dedup_merges_overlapping_facts(monkeypatch, tmp_path):
    mem = _redirect_memory_paths(monkeypatch, tmp_path)
    summary = mem.empty_memory_skeleton()
    summary["shared_facts"] = ["The project uses Ollama."]

    comp = SimpleNamespace(
        shared_facts=["The project uses Ollama for local inference."],
        decisions_log=[],
        persona_notes={},
        open_questions=[],
        reasoning="dup test",
    )
    _install_compactor_mock(monkeypatch, mem, [comp])

    merged = mem.compact([{"role": "user", "content": "it's local"}], summary)
    facts = merged["shared_facts"]
    # Only one fact remains.
    assert len(facts) == 1
    # And the more specific phrasing wins.
    assert "local inference" in facts[0]


def test_dedup_drops_pure_duplicate():
    from src.agents.memory import _merge_string_list
    out = _merge_string_list(
        existing=["The project uses Ollama."],
        incoming=["The project uses Ollama."],
    )
    assert out == ["The project uses Ollama."]


def test_compact_accepts_fenced_json_from_provider(monkeypatch):
    from src.agents import memory as mem

    fenced_response = """```json
{
  "shared_facts": ["The service uses Redis."],
  "decisions_log": [],
  "persona_notes": {},
  "open_questions": [],
  "reasoning": "Captured a supported fact."
}
```"""
    _install_compactor_mock(monkeypatch, mem, [fenced_response])

    merged = mem.compact([], mem.empty_memory_skeleton())
    assert merged["shared_facts"] == ["The service uses Redis."]


# Test 7 — Corrupted JSON
def test_corrupted_json_is_quarantined(monkeypatch, tmp_path):
    mem = _redirect_memory_paths(monkeypatch, tmp_path)
    bad = tmp_path / "sessions" / "rolling_summary.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("not valid json {{{", encoding="utf-8")

    summary = mem.load_session()
    # Empty skeleton is returned.
    assert summary["shared_facts"] == []
    assert summary["decisions_log"] == []
    # The bad file should NOT have been overwritten by a fresh empty
    # summary; the original bytes must remain on disk, and a quarantine
    # file should have been created.
    assert bad.read_text(encoding="utf-8") == "not valid json {{{"
    quarantined = list((tmp_path / "sessions").glob("rolling_summary.json.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "not valid json {{{"


# ── Extra: defensive coverage of helpers ────────────────────────────

def test_split_window_invariants():
    from src.agents.memory import split_window, make_turn
    buf = [make_turn("user", str(i)) for i in range(7)]
    old, recent = split_window(buf, window_size=4)
    assert len(recent) == 4
    assert len(old) == 3
    # Reassembly invariant
    assert old + recent == buf


def test_compact_persona_bucket_injection_dropped():
    """If the LLM tries to invent a new persona bucket, merge must drop it."""
    from src.agents.memory import MemoryCompactionResult, merge_compaction
    existing = {
        "last_updated": "",
        "shared_facts": [],
        "decisions_log": [],
        "persona_notes": {"capability_maximizer": [], "practical_architect": []},
        "open_questions": [],
    }
    comp = MemoryCompactionResult(
        shared_facts=[],
        decisions_log=[],
        persona_notes={
            "capability_maximizer": ["ok"],
            "rogue_new_persona": ["hacker note"],
        },
        open_questions=[],
        reasoning="",
    )
    merged = merge_compaction(existing, comp)
    assert "rogue_new_persona" not in merged["persona_notes"]
    assert merged["persona_notes"]["capability_maximizer"] == ["ok"]
