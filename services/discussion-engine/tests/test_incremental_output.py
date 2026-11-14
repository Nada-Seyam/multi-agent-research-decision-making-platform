from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from discussion import orchestrator
from discussion.conclusion import ConclusionQualityWarning
from discussion.artifacts import (
    FileDiscussionArtifactStore,
    NullDiscussionArtifactStore,
)
from discussion.config import DiscussionSettings
from discussion.delivery import replay_pending_events
from discussion.event_sinks import DeliveryResult, EventDeliveryError, HttpEventSink
from discussion.events import DiscussionEventType, make_discussion_event
from discussion.models import SharedResearchContext

PERSONAS = {
    "agent_a": {},
    "agent_b": {},
    "agent_c": {},
    "agent_d": {},
}


class RecordingStore(NullDiscussionArtifactStore):
    def __init__(self):
        self.snapshots = []
        self.events = []
        self.deliveries = []

    def save_snapshot(self, snapshot):
        self.snapshots.append(snapshot)

    def append_event(self, event):
        self.events.append(event)

    def record_delivery(self, event_id, **record):
        self.deliveries.append((event_id, record))


class RecordingSink:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)
        return DeliveryResult(
            status="delivered",
            attempts=1,
            attempted_at="2026-09-18T00:00:00Z",
            status_code=201,
        )


class FailFirstRoundSink(RecordingSink):
    def publish(self, event):
        self.events.append(event)
        if event.event_type is DiscussionEventType.ROUND_COMPLETED:
            return DeliveryResult(
                status="pending",
                attempts=1,
                attempted_at="2026-09-18T00:00:00Z",
                last_error="consumer unavailable",
                status_code=503,
            )
        return DeliveryResult(
            status="delivered",
            attempts=1,
            attempted_at="2026-09-18T00:00:00Z",
            status_code=201,
        )


def _settings(**overrides):
    values = {
        "num_participants": 4,
        "num_rounds": 3,
        "random_seed": 42,
        "week2_root": "week2",
        "persist_artifacts": False,
    }
    values.update(overrides)
    return DiscussionSettings(**values)


def _objective():
    return SimpleNamespace(
        goal="Choose an architecture",
        short_id=lambda: "choose_an_architecture",
    )


def _install_fakes(monkeypatch):
    research = SharedResearchContext(
        plan=SimpleNamespace(queries=["q"]),
        evidence_pool=[],
        dossier=SimpleNamespace(summary="dossier"),
    )
    monkeypatch.setattr(
        orchestrator.week2_adapter, "load_week2_personas", lambda root: PERSONAS
    )
    monkeypatch.setattr(
        orchestrator, "build_shared_research", lambda objective, **kwargs: research
    )
    monkeypatch.setattr(
        orchestrator,
        "generate_conclusion",
        lambda opinions: "Final conclusion from final-round opinions.",
    )
    monkeypatch.setattr(
        orchestrator.week2_adapter,
        "run_for_persona",
        lambda persona_id, objective, dossier, *, memory_context=None: {
            "opinion": f"Opinion from {persona_id}\nPOSITION: candidate_{persona_id}"
        },
    )


def test_orchestrator_persists_and_emits_each_completed_round(monkeypatch):
    _install_fakes(monkeypatch)
    store = RecordingStore()
    sink = RecordingSink()

    result = orchestrator.run_discussion(
        _objective(),
        settings=_settings(),
        artifact_store=store,
        event_sink=sink,
    )

    assert [event.event_type for event in store.events] == [
        DiscussionEventType.STARTED,
        DiscussionEventType.ROUND_COMPLETED,
        DiscussionEventType.ROUND_COMPLETED,
        DiscussionEventType.ROUND_COMPLETED,
        DiscussionEventType.COMPLETED,
    ]
    assert [snapshot["status"] for snapshot in store.snapshots] == [
        "preparing",
        "running",
        "running",
        "running",
        "completed",
    ]
    assert [len(snapshot["rounds"]) for snapshot in store.snapshots] == [0, 1, 2, 3, 3]
    assert store.events[1].data["round"] == store.snapshots[1]["rounds"][0]
    assert sink.events == store.events
    assert result.status == "completed"
    assert result.decision_summary is not None
    assert result.conclusion == "Final conclusion from final-round opinions."


def test_orchestrator_checkpoints_research_preparation_without_extra_events(monkeypatch):
    _install_fakes(monkeypatch)
    store = RecordingStore()
    sink = RecordingSink()
    research = SharedResearchContext(
        plan=SimpleNamespace(queries=["q"]),
        evidence_pool=[],
        dossier=SimpleNamespace(summary="dossier"),
    )

    def fake_research(objective, *, settings=None, progress_callback=None):
        for stage in ("planning", "retrieving", "ready"):
            progress_callback(
                {
                    "stage": stage,
                    "message": stage,
                    "completed_units": int(stage == "ready"),
                    "total_units": 1,
                    "queries": [],
                }
            )
        return research

    monkeypatch.setattr(orchestrator, "build_shared_research", fake_research)

    orchestrator.run_discussion(
        _objective(),
        settings=_settings(num_rounds=3),
        artifact_store=store,
        event_sink=sink,
    )

    assert [snapshot["preparation"]["stage"] for snapshot in store.snapshots[:4]] == [
        "room",
        "planning",
        "retrieving",
        "ready",
    ]
    assert [event.event_type for event in store.events] == [
        DiscussionEventType.STARTED,
        DiscussionEventType.ROUND_COMPLETED,
        DiscussionEventType.ROUND_COMPLETED,
        DiscussionEventType.ROUND_COMPLETED,
        DiscussionEventType.COMPLETED,
    ]


def test_required_delivery_failure_preserves_round_and_emits_failed(monkeypatch):
    _install_fakes(monkeypatch)
    store = RecordingStore()
    sink = FailFirstRoundSink()

    with pytest.raises(EventDeliveryError):
        orchestrator.run_discussion(
            _objective(),
            settings=_settings(event_delivery_required=True),
            artifact_store=store,
            event_sink=sink,
        )

    assert [snapshot["status"] for snapshot in store.snapshots] == [
        "preparing",
        "running",
        "failed",
    ]
    assert len(store.snapshots[-1]["rounds"]) == 1
    assert store.snapshots[-1]["failure"]["stage"] == "round_1"
    assert store.snapshots[-1]["failure"]["round"] == 1
    assert [event.event_type for event in store.events] == [
        DiscussionEventType.STARTED,
        DiscussionEventType.ROUND_COMPLETED,
        DiscussionEventType.FAILED,
    ]


def test_conclusion_quality_issue_completes_with_warning(monkeypatch):
    _install_fakes(monkeypatch)
    store = RecordingStore()
    sink = RecordingSink()
    monkeypatch.setattr(
        orchestrator,
        "generate_conclusion",
        lambda opinions: (_ for _ in ()).throw(
            ConclusionQualityWarning(
                ["states a technical claim with prohibited absolute language"],
                "Recommendations\n\nStructured fallback conclusion.",
            )
        ),
    )

    result = orchestrator.run_discussion(
        _objective(),
        settings=_settings(),
        artifact_store=store,
        event_sink=sink,
    )

    assert result.status == "completed"
    assert result.failure is None
    assert result.conclusion == "Recommendations\n\nStructured fallback conclusion."
    assert result.warnings[0]["code"] == "conclusion_quality"
    assert store.snapshots[-1]["status"] == "completed"
    assert store.snapshots[-1]["warnings"][0]["stage"] == "conclusion"
    assert store.events[-1].event_type is DiscussionEventType.COMPLETED


def test_event_token_is_not_serialized(monkeypatch):
    _install_fakes(monkeypatch)
    store = RecordingStore()

    orchestrator.run_discussion(
        _objective(),
        settings=_settings(event_token="top-secret"),
        artifact_store=store,
        event_sink=RecordingSink(),
    )

    assert "event_token" not in store.snapshots[-1]["configuration"]
    assert "top-secret" not in json.dumps(store.snapshots[-1], default=str)


def test_file_store_writes_valid_snapshot_and_replays_pending_event(tmp_path):
    store = FileDiscussionArtifactStore(tmp_path / "outputs", tmp_path / "logs")
    snapshot = {
        "schema_version": "1.0",
        "discussion_id": "discussion-1",
        "rounds": [],
    }
    event = make_discussion_event(DiscussionEventType.STARTED, snapshot, sequence=0)
    store.save_snapshot(snapshot)
    store.append_event(event)

    assert (
        json.loads(store.snapshot_path("discussion-1").read_text(encoding="utf-8"))
        == snapshot
    )
    assert [item.event_id for item in store.pending_events("discussion-1")] == [
        event.event_id
    ]

    sink = RecordingSink()
    summary = replay_pending_events("discussion-1", store, sink)
    assert summary.attempted == 1
    assert summary.delivered == 1
    assert summary.pending == 0
    assert store.pending_events("discussion-1") == []


def test_skipped_events_remain_available_for_later_replay(tmp_path):
    store = FileDiscussionArtifactStore(tmp_path / "outputs", tmp_path / "logs")
    snapshot = {"discussion_id": "discussion-1", "rounds": []}
    event = make_discussion_event(DiscussionEventType.STARTED, snapshot, sequence=0)
    store.append_event(event)
    store.record_delivery(
        event.event_id,
        status="skipped",
        attempts=0,
        last_error=None,
        status_code=None,
        attempted_at="2026-09-18T00:00:00Z",
    )

    assert [item.event_id for item in store.pending_events("discussion-1")] == [
        event.event_id
    ]


def test_http_sink_retries_retryable_response_and_uses_idempotency_key():
    attempts = []

    def handler(request):
        attempts.append(request)
        status = 503 if len(attempts) == 1 else 201
        return httpx.Response(status, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sink = HttpEventSink(
        "http://consumer.test/api/v1/discussion-events",
        token="secret",
        max_attempts=2,
        retry_initial_seconds=0,
        client=client,
    )
    snapshot = {"discussion_id": "discussion-1", "rounds": []}
    event = make_discussion_event(DiscussionEventType.STARTED, snapshot, sequence=0)

    result = sink.publish(event)

    assert result.delivered is True
    assert result.attempts == 2
    assert attempts[0].headers["Idempotency-Key"] == event.event_id
    assert attempts[0].headers["Authorization"] == "Bearer secret"
    client.close()


def test_http_sink_reuses_its_owned_client(monkeypatch):
    requests = []
    clients = []
    real_client = httpx.Client

    def handler(request):
        requests.append(request)
        return httpx.Response(201, request=request)

    def client_factory(*, timeout):
        client = real_client(transport=httpx.MockTransport(handler), timeout=timeout)
        clients.append(client)
        return client

    monkeypatch.setattr("discussion.event_sinks.httpx.Client", client_factory)
    sink = HttpEventSink("http://consumer.test/events")
    snapshot = {"discussion_id": "discussion-1", "rounds": []}

    sink.publish(make_discussion_event(DiscussionEventType.STARTED, snapshot, sequence=0))
    sink.publish(make_discussion_event(DiscussionEventType.COMPLETED, snapshot, sequence=1))
    sink.close()

    assert len(clients) == 1
    assert len(requests) == 2
    assert clients[0].is_closed
