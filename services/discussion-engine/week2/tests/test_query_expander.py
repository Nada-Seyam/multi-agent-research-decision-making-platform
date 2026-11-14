"""Unit tests for the query expander.

Mocks the LLM client so the test runs without Ollama or any network.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))


def _install_mock_llm(monkeypatch, responses):
    """Patch ``get_llm`` on the query_expander module to return canned outputs."""
    from src.agents import query_expander as qe_mod
    import json

    class _FakeResponse:
        def __init__(self, payload):
            self.content = payload

    class _FakeLLM:
        def __init__(self, queue):
            self._queue = list(queue)

        def invoke(self, _messages):
            ns = self._queue.pop(0)
            payload = json.dumps({"reasoning": ns.reasoning, "expanded_query": getattr(ns, "expanded_query", "")})
            return _FakeResponse(payload)

    queue = list(responses)
    monkeypatch.setattr(qe_mod, "get_llm", lambda: _FakeLLM(queue))


def test_expand_query_returns_rewritten_string(monkeypatch):
    """Happy path: structured output returns an expanded_query field."""
    _install_mock_llm(monkeypatch, [
        SimpleNamespace(reasoning="missing acronym", expanded_query="INT4 AWQ quantization benchmark")
    ])

    from src.agents.query_expander import expand_query

    out = expand_query(
        query="INT4 quantization edge",
        chunks=[{"embedding_text": "FP8 paper", "source_name": "p1"}],
        grading_reasoning="retrieved FP8 not INT4",
    )
    assert out == "INT4 AWQ quantization benchmark"


def test_expand_query_falls_back_on_empty_response(monkeypatch):
    """If the LLM returns a payload with an empty expanded_query, keep the original."""
    _install_mock_llm(monkeypatch, [
        SimpleNamespace(reasoning="...", expanded_query="")
    ])

    from src.agents.query_expander import expand_query

    out = expand_query("orig", [{"text": "x"}], "irrelevant")
    assert out == "orig"


def test_expand_query_falls_back_on_exception(monkeypatch):
    """If invoke throws, the original query is returned."""
    from src.agents import query_expander as qe_mod

    class _BoomLLM:
        def invoke(self, _messages):
            raise RuntimeError("LLM down")

    monkeypatch.setattr(qe_mod, "get_llm", lambda: _BoomLLM())

    from src.agents.query_expander import expand_query

    out = expand_query("orig", [], "no chunks")
    assert out == "orig"


def test_expand_query_empty_input_returns_empty():
    """Defensive: empty query is returned untouched (no LLM call)."""
    from src.agents.query_expander import expand_query
    assert expand_query("", [], "reason") == ""
    assert expand_query("   ", [], "reason") == "   "


def test_expand_query_chunks_truncated_for_preview(monkeypatch):
    """Long chunk text is truncated so the prompt stays bounded."""
    captured = {}
    import json

    class _FakeResponse:
        def __init__(self, payload):
            self.content = payload

    class _CapturingLLM:
        def invoke(self, messages):
            captured["messages"] = messages
            payload = json.dumps({"reasoning": "ok", "expanded_query": "rewritten"})
            return _FakeResponse(payload)

    from src.agents import query_expander as qe_mod
    monkeypatch.setattr(qe_mod, "get_llm", lambda: _CapturingLLM())

    from src.agents.query_expander import expand_query

    long_text = "x" * 5000
    expand_query(
        "q",
        [{"embedding_text": long_text, "source_name": "src"}],
        "reason",
    )
    # The Human message contains the chunk preview; verify truncation kicked in.
    human = next(m for m in captured["messages"] if m.__class__.__name__ == "HumanMessage")
    assert "..." in human.content
    # The original 5000-char text is not present verbatim.
    assert long_text not in human.content
