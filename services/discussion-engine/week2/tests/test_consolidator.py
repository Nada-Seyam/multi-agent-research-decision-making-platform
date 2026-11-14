"""Unit tests for the evidence consolidator.

All tests run offline. LLM calls are monkeypatched at the ``_client`` import site.
"""
from __future__ import annotations

import sys
from pathlib import Path

_FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
if str(_FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))

from src.agents.objective import CoreTopic, Objective
from src.agents.resolve_context import ContextResolutionResult
from src.agents.evidence_pool import EvidenceResult
from src.agents.consolidator import (
    ResearchDossier,
    consolidate_evidence,
    _dedup_by_text,
    _best_topic,
    _decode_first_json,
    _parse_disagreements,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_objective() -> Objective:
    return Objective(
        goal="Build a Transformer from scratch.",
        constraints=["Limited compute"],
        core_topics=[
            CoreTopic(id="attention", title="ATTENTION", subpoints=["MQA vs GQA", "sliding window"]),
            CoreTopic(id="depth", title="DEPTH", subpoints=["deep vs shallow"]),
            CoreTopic(id="ffn", title="FFN", subpoints=["dense vs MoE", "routing"]),
        ],
        evidence_requirements=["Always cite"],
        decision_criteria=["Quality", "Compute"],
        strict_notes=["No speculation"],
        final_goal="Produce a practical design.",
    )


def _make_er(
    query: str,
    chunks: list[dict],
    order: int = 0,
    *,
    source_used: str = "rag",
    is_relevant: bool = True,
) -> EvidenceResult:
    return EvidenceResult(
        query=query,
        context=ContextResolutionResult(
            chunks=chunks,
            resolved_query=query,
            is_relevant=is_relevant,
            source_used=source_used,
            expansion_attempts=0,
        ),
        order=order,
    )


def _chunk(text: str, source: str = "paper.pdf", **kw) -> dict:
    return {"embedding_text": text, "source_name": source, **kw}


# ── Tests: dedup ────────────────────────────────────────────────────


class TestDedupByText:
    def test_collapses_exact_duplicates(self):
        items = [
            {"citation_index": 1, "text": "MoE is efficient", "source_name": "p1",
             "source": "rag", "scores": {}, "research_query_id": 1,
             "research_question": "Q", "final_query": "Q", "is_relevant": True,
             "section": "", "chunk_id": ""},
            {"citation_index": 2, "text": "MoE is efficient", "source_name": "p2",
             "source": "web", "scores": {"rerank_score": 0.9}, "research_query_id": 2,
             "research_question": "Q2", "final_query": "Q2", "is_relevant": True,
             "section": "", "chunk_id": ""},
        ]
        result = _dedup_by_text(items)
        assert len(result) == 1
        # Should keep the one with richer metadata (web with rerank_score).
        assert result[0]["source_name"] == "p2"
        assert result[0]["scores"]["rerank_score"] == 0.9

    def test_preserves_different_text(self):
        items = [
            {"citation_index": 1, "text": "MoE is efficient", "source_name": "p1",
             "source": "rag", "scores": {}, "research_query_id": 1,
             "research_question": "Q", "final_query": "Q", "is_relevant": True,
             "section": "", "chunk_id": ""},
            {"citation_index": 2, "text": "Dense outperforms MoE", "source_name": "p2",
             "source": "rag", "scores": {}, "research_query_id": 2,
             "research_question": "Q2", "final_query": "Q2", "is_relevant": True,
             "section": "", "chunk_id": ""},
        ]
        result = _dedup_by_text(items)
        assert len(result) == 2

    def test_keeps_rag_when_scores_tied(self):
        """When both duplicates have empty scores, rag wins over web."""
        items = [
            {"citation_index": 1, "text": "same", "source_name": "p1",
             "source": "rag", "scores": {}, "research_query_id": 1,
             "research_question": "Q", "final_query": "Q", "is_relevant": True,
             "section": "", "chunk_id": ""},
            {"citation_index": 2, "text": "same", "source_name": "p2",
             "source": "web", "scores": {}, "research_query_id": 2,
             "research_question": "Q2", "final_query": "Q2", "is_relevant": True,
             "section": "", "chunk_id": ""},
        ]
        result = _dedup_by_text(items)
        assert len(result) == 1
        assert result[0]["source_name"] == "p1"


# ── Tests: topic assignment ─────────────────────────────────────────


class TestBestTopic:
    def test_assigns_to_matching_topic(self):
        objective = _make_objective()
        from src.agents.consolidator import _topic_map
        tm = _topic_map(objective)
        assert _best_topic("Multi-head attention vs grouped-query attention", tm) == "attention"
        assert _best_topic("Shallow vs deep transformer layers", tm) == "depth"

    def test_returns_none_for_unrelated_text(self):
        objective = _make_objective()
        from src.agents.consolidator import _topic_map
        tm = _topic_map(objective)
        assert _best_topic("The weather is nice today", tm) is None


def test_disagreement_parser_keeps_valid_items_and_skips_invalid_items():
    data = _decode_first_json(
        'Here is the result: {"disagreements": ['
        '{"excerpt_indices": [1, 2], "description": "valid"},'
        '{"excerpt_indices": [18], "description": "not a pair"}'
        ']} trailing text'
    )

    contradictions = _parse_disagreements(data, max_index=2)

    assert len(contradictions) == 1
    assert contradictions[0].excerpt_indices == [1, 2]


# ── Tests: consolidate_evidence with fixture pools ──────────────────


class TestConsolidateEvidence:
    def test_empty_pool_returns_empty_dossier(self):
        dossier = consolidate_evidence([], _make_objective())
        assert isinstance(dossier, ResearchDossier)
        assert len(dossier.excerpts) == 0
        assert dossier.objective_id

    def test_rejected_chunks_do_not_enter_dossier(self, monkeypatch):
        pool = [
            _make_er(
                "Q1",
                [
                    _chunk("Rejected context", "bad.pdf", accepted=False),
                    _chunk("Accepted context", "good.pdf", accepted=True),
                ],
            )
        ]
        _stub_llm(monkeypatch)

        dossier = consolidate_evidence(pool, _make_objective())

        assert [excerpt.text for excerpt in dossier.excerpts] == ["Accepted context"]

    def test_null_chunk_metadata_is_normalized(self, monkeypatch):
        pool = [
            _make_er(
                "Q1",
                [_chunk("Evidence with missing section", section=None, chunk_id=None)],
            )
        ]
        _stub_llm(monkeypatch)

        dossier = consolidate_evidence(pool, _make_objective())

        assert dossier.excerpts[0].section == ""
        assert dossier.excerpts[0].chunk_id == ""

    def test_dedup_collapses_duplicate_claims(self, monkeypatch):
        """Two chunks with identical text from different sources collapse to one excerpt."""
        pool = [
            _make_er("Q1", [_chunk("MoE is better for efficiency", "p1.pdf")], order=0),
            _make_er("Q2", [_chunk("MoE is better for efficiency", "p2.pdf")], order=1),
        ]
        _stub_llm(monkeypatch)
        dossier = consolidate_evidence(pool, _make_objective())
        assert len(dossier.excerpts) == 1
        assert dossier.excerpts[0].evidence_tier in ("F", "I", "S")

    def test_evidence_tier_tag_present_on_every_excerpt(self, monkeypatch):
        pool = [
            _make_er("Q1", [
                _chunk("GQA reduces KV-cache by 4x", "p1.pdf"),
                _chunk("MoE is speculative for small models", "p2.pdf"),
            ], order=0),
        ]
        _stub_llm(monkeypatch)
        dossier = consolidate_evidence(pool, _make_objective())
        for ex in dossier.excerpts:
            assert ex.evidence_tier in ("F", "I", "S")

    def test_source_disagreement_flagged(self, monkeypatch):
        """Two excerpts with opposing claims produce a contradiction in the dossier."""
        pool = [
            _make_er("Q1", [
                _chunk("MoE is more efficient than dense for inference", "paper1.pdf"),
                _chunk("Dense models outperform MoE on quality", "paper2.pdf"),
            ], order=0),
        ]

        # Stub LLM: tier tags both as F, detects disagreement
        class _FakeTierResponse:
            content = '{"tags": [{"index": 1, "evidence_tier": "F"}, {"index": 2, "evidence_tier": "F"}]}'

        class _FakeDisagreementResponse:
            content = (
                '{"disagreements": [{"excerpt_indices": [1, 2], '
                '"description": "Opposing claims about MoE vs Dense efficiency.", '
                '"topic_id": "ffn"}]}'
            )

        call_count = [0]

        class _FakeLLM:
            def invoke(self, messages):
                call_count[0] += 1
                if call_count[0] == 1:
                    return _FakeTierResponse()
                return _FakeDisagreementResponse()

        from src.agents import consolidator as cons_mod
        monkeypatch.setattr(cons_mod, "get_llm", lambda: _FakeLLM())

        dossier = consolidate_evidence(pool, _make_objective())
        assert len(dossier.contradictions) == 1
        c = dossier.contradictions[0]
        assert 1 in c.excerpt_indices
        assert 2 in c.excerpt_indices
        assert "MoE" in c.description or "Dense" in c.description

    def test_tier_tagging_mechanical_fallback(self, monkeypatch):
        """When LLM raises, all excerpts default to tier I."""
        pool = [
            _make_er("Q1", [_chunk("Some evidence", "p1.pdf")], order=0),
        ]
        from src.agents import consolidator as cons_mod
        def _raise(*_a, **_k):
            raise RuntimeError("LLM down")
        monkeypatch.setattr(cons_mod, "get_llm", lambda: _raise)

        dossier = consolidate_evidence(pool, _make_objective())
        assert len(dossier.excerpts) == 1
        assert dossier.excerpts[0].evidence_tier == "I"

    def test_per_topic_summaries_populated(self, monkeypatch):
        pool = [
            _make_er("Q1", [
                _chunk("GQA attention reduces KV cache", "p1.pdf"),
                _chunk("Sliding window attention limits context", "p2.pdf"),
            ], order=0),
            _make_er("Q2", [_chunk("Deep transformers are harder to train", "p3.pdf")], order=1),
        ]
        _stub_llm(monkeypatch)
        dossier = consolidate_evidence(pool, _make_objective())
        assert len(dossier.per_topic_summaries) >= 1

    def test_dossier_summary_is_string(self, monkeypatch):
        pool = [_make_er("Q1", [_chunk("Evidence text", "p1.pdf")], order=0)]
        _stub_llm(monkeypatch)
        dossier = consolidate_evidence(pool, _make_objective())
        assert isinstance(dossier.summary, str)
        assert "RESEARCH DOSSIER" in dossier.summary

    def test_dossier_preserves_transfer_limits_and_missing_coverage(self, monkeypatch):
        pool = [
            _make_er(
                "Compare architectures under target constraints",
                [_chunk(
                    "A small-scale benchmark reports a throughput difference.",
                    "p1.pdf",
                    applicability={
                        "model_scale": "low",
                        "hardware": "unknown",
                        "training_stage": "high",
                        "metric": "medium",
                    },
                    limitations=["The reported model scale is below the target range."],
                )],
            )
        ]
        _stub_llm(monkeypatch)

        dossier = consolidate_evidence(pool, _make_objective())

        assert dossier.excerpts[0].research_query.startswith("Compare architectures")
        assert dossier.excerpts[0].limitations == [
            "The reported model scale is below the target range."
        ]
        assert dossier.coverage["attention"]["status"] in {"partial", "missing"}
        assert dossier.unresolved_requirements
        assert "Neutrality Notice" in dossier.summary
        assert "Transferability Limits" in dossier.summary
        assert "Unresolved" in dossier.summary


# ── LLM stub helper ────────────────────────────────────────────────


def _stub_llm(monkeypatch, *, tier_map=None):
    """Install a fake LLM that returns sensible defaults."""
    from src.agents import consolidator as cons_mod
    tier_map = tier_map or {}

    class _FakeTierResponse:
        def __init__(self, count):
            tags = [{"index": i + 1, "evidence_tier": tier_map.get(i + 1, "I")} for i in range(count)]
            import json
            self.content = json.dumps({"tags": tags})

    class _FakeDisagreementResponse:
        content = '{"disagreements": []}'

    class _FakeLLM:
        def __init__(self):
            self._call = 0
        def invoke(self, messages):
            self._call += 1
            # First call is tier tagging, second is disagreement detection.
            if self._call == 1:
                import re as _re
                human_msg = messages[-1].content if messages else ""
                indices = _re.findall(r"\[(\d+)\]", human_msg)
                max_idx = max((int(i) for i in indices), default=0)
                return _FakeTierResponse(max_idx)
            return _FakeDisagreementResponse()

    monkeypatch.setattr(cons_mod, "get_llm", lambda: _FakeLLM())
