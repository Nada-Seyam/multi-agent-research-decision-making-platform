from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from analytics import streaming_service as streaming_service_module
from analytics.data_loader import DiscussionValidationError, load_discussion
from analytics.engine import analyze_round
from analytics.streaming_events import canonical_sha256
from analytics.streaming_service import StreamingAnalyticsService
from analytics.streaming_store import AnalyticsArtifactStore


def _snapshot(round_count: int, status: str = "running"):
    rounds = []
    for round_number in range(1, round_count + 1):
        rounds.append(
            {
                "round": round_number,
                "opinions": [
                    {"persona_id": "a", "architecture": "x", "confidence": 0.8},
                    {"persona_id": "b", "architecture": "x", "confidence": 0.7},
                ],
                "messages": [],
            }
        )
    return {
        "schema_version": "1.0",
        "discussion_id": "streaming-api-test",
        "status": status,
        "participants": ["a", "b"],
        "rounds": rounds,
        "decision_summary": (
            {"preferred_candidate": "x", "consensus_status": "consensus"}
            if status == "completed"
            else None
        ),
        "objective": {"goal": "Choose x"},
        "graph": {"agents": ["a", "b"], "edges": [["a", "b"]], "directed": True},
        "configuration": {},
    }


def _event(event_type, sequence, snapshot, completed_round=None):
    suffix = (
        f"round:{completed_round}"
        if event_type == "round.completed"
        else event_type.split(".", 1)[1]
    )
    data = {"discussion": snapshot}
    if completed_round is not None:
        data["completed_round"] = completed_round
        data["round"] = snapshot["rounds"][completed_round - 1]
    return {
        "schema_version": "1.0",
        "event_id": f"streaming-api-test:{suffix}",
        "event_type": event_type,
        "discussion_id": "streaming-api-test",
        "sequence": sequence,
        "occurred_at": "2026-01-01T00:00:00Z",
        "snapshot_sha256": canonical_sha256(snapshot),
        "data": data,
    }


def test_round_analysis_defers_influence():
    result = analyze_round(load_discussion(_snapshot(1)), 1)

    assert result.round == 1
    assert "influence" in result.deferred_metrics
    assert not hasattr(result, "influence")


def test_streaming_service_emits_round_and_final_analysis(tmp_path):
    store = AnalyticsArtifactStore(tmp_path / "outputs", tmp_path / "logs")
    service = StreamingAnalyticsService(store)
    round_event = _event("round.completed", 1, _snapshot(1), 1)
    completed_event = _event(
        "discussion.completed", 2, _snapshot(1, "completed")
    )

    round_result = service.process(round_event, round_event["event_id"])
    final_result = service.process(completed_event, completed_event["event_id"])
    artifact = store.read_artifact("streaming-api-test")

    assert round_result.analysis_event["event_type"] == "analysis.round.completed"
    assert final_result.analysis_event["event_type"] == (
        "analysis.discussion.completed"
    )
    assert artifact["status"] == "completed"
    assert "influence" not in artifact["round_analyses"][0]
    assert set(artifact["final_analysis"]["influence"]) == {"a", "b"}
    assert (
        store.logs_dir
        / "streaming-api-test"
        / "events"
        / "round-01-completed-source-event.json"
    ).exists()


def test_ingest_is_durable_before_expensive_processing(tmp_path):
    store = AnalyticsArtifactStore(tmp_path / "outputs", tmp_path / "logs")
    service = StreamingAnalyticsService(store)
    event = _event("round.completed", 1, _snapshot(1), 1)

    accepted = service.ingest(event, event["event_id"])

    assert accepted.status == "accepted"
    assert accepted.should_process is True
    assert store.read_event("streaming-api-test", event["event_id"]) == event
    assert store.read_artifact("streaming-api-test") is None

    service.process_ingested(event, event["event_id"])
    artifact = store.read_artifact("streaming-api-test")
    assert artifact["processed_source_events"] == [event["event_id"]]


def test_pending_events_are_recovered_after_restart(tmp_path):
    store = AnalyticsArtifactStore(tmp_path / "outputs", tmp_path / "logs")
    event = _event("round.completed", 1, _snapshot(1), 1)
    StreamingAnalyticsService(store).ingest(event, event["event_id"])

    recovered = StreamingAnalyticsService(store).recover_pending()

    assert recovered == 1
    artifact = store.read_artifact("streaming-api-test")
    assert artifact["processed_source_events"] == [event["event_id"]]


def test_background_processing_failure_is_visible_and_retryable(
    tmp_path, monkeypatch
):
    store = AnalyticsArtifactStore(tmp_path / "outputs", tmp_path / "logs")
    service = StreamingAnalyticsService(store)
    event = _event("round.completed", 1, _snapshot(1), 1)
    service.ingest(event, event["event_id"])
    original = streaming_service_module.analyze_round

    def fail_analysis(*_args, **_kwargs):
        raise RuntimeError("inference backend unavailable")

    monkeypatch.setattr(
        streaming_service_module,
        "analyze_round",
        fail_analysis,
    )
    result = service.process_ingested_safely(event, event["event_id"])
    failed_artifact = store.read_artifact("streaming-api-test")

    assert result.status == "processing_failed"
    assert failed_artifact["status"] == "analysis_failed"
    assert failed_artifact["processing_failures"][event["event_id"]][
        "error_type"
    ] == "RuntimeError"
    assert event["event_id"] not in failed_artifact["processed_source_events"]

    monkeypatch.setattr(streaming_service_module, "analyze_round", original)
    assert service.recover_pending() == 1
    recovered_artifact = store.read_artifact("streaming-api-test")
    assert recovered_artifact["status"] == "running"
    assert recovered_artifact["processing_failures"] == {}
    assert recovered_artifact["processed_source_events"] == [event["event_id"]]


def test_late_round_is_reanchored_to_already_completed_decision(tmp_path):
    store = AnalyticsArtifactStore(tmp_path / "outputs", tmp_path / "logs")
    service = StreamingAnalyticsService(store)
    completed_snapshot = _snapshot(1, "completed")
    completed_snapshot["decision_summary"]["preferred_candidate"] = "y"
    completed_event = _event("discussion.completed", 2, completed_snapshot)
    round_event = _event("round.completed", 1, _snapshot(1), 1)

    service.process(completed_event, completed_event["event_id"])
    service.process(round_event, round_event["event_id"])

    artifact = store.read_artifact("streaming-api-test")
    round_analysis = artifact["round_analyses"][0]
    assert artifact["status"] == "completed"
    assert round_analysis["anchor_candidate"] == "y"
    assert round_analysis["anchor_source"] == "decision_summary"
    assert round_analysis["provisional_anchor"] is False
    assert {record["stance"] for record in round_analysis["stances"].values()} == {-1.0}


def test_ingestion_does_not_wait_for_same_discussion_inference(
    tmp_path, monkeypatch
):
    store = AnalyticsArtifactStore(tmp_path / "outputs", tmp_path / "logs")
    service = StreamingAnalyticsService(store)
    round_event = _event("round.completed", 1, _snapshot(1), 1)
    completed_event = _event(
        "discussion.completed", 2, _snapshot(1, "completed")
    )
    service.ingest(round_event, round_event["event_id"])
    inference_started = Event()
    release_inference = Event()
    original = streaming_service_module.analyze_round

    def slow_analysis(*args, **kwargs):
        inference_started.set()
        release_inference.wait(timeout=2)
        return original(*args, **kwargs)

    monkeypatch.setattr(streaming_service_module, "analyze_round", slow_analysis)
    with ThreadPoolExecutor(max_workers=2) as executor:
        processing = executor.submit(
            service.process_ingested,
            round_event,
            round_event["event_id"],
        )
        assert inference_started.wait(timeout=1)
        ingestion = executor.submit(
            service.ingest,
            completed_event,
            completed_event["event_id"],
        )
        try:
            accepted = ingestion.result(timeout=0.5)
        finally:
            release_inference.set()
        processing.result(timeout=2)

    assert accepted.status == "accepted"


def test_ingest_rejects_invalid_analysis_snapshot_before_acknowledging(tmp_path):
    store = AnalyticsArtifactStore(tmp_path / "outputs", tmp_path / "logs")
    service = StreamingAnalyticsService(store)
    event = _event("round.completed", 1, _snapshot(1), 1)
    invalid_snapshot = event["data"]["discussion"]
    invalid_snapshot["rounds"] = []
    event["snapshot_sha256"] = canonical_sha256(invalid_snapshot)

    with pytest.raises(DiscussionValidationError):
        service.ingest(event, event["event_id"])

    assert store.read_event("streaming-api-test", event["event_id"]) is None
