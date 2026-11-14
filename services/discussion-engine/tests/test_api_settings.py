import json
import asyncio
from io import BytesIO
from types import SimpleNamespace

from fastapi import BackgroundTasks, UploadFile
import pytest
from starlette.requests import Request

from discussion import api, config
from discussion.artifacts import FileDiscussionArtifactStore
from discussion.config import DiscussionSettings
from discussion.events import DiscussionEventType, make_discussion_event


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/discussions",
            "raw_path": b"/api/v1/discussions",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8001),
        }
    )


def test_local_api_uses_configured_consumer(monkeypatch):
    configured = DiscussionSettings(
        random_seed=1,
        event_endpoint="http://127.0.0.1:8002/api/v1/discussion-events",
    )
    monkeypatch.setattr(api, "get_settings", lambda: configured)

    result = api._settings_for_local_api(_request())

    assert result.event_endpoint == configured.event_endpoint
    assert result.event_delivery_required is True
    assert result.persist_artifacts is True


def test_browser_payload_builds_a_valid_week2_objective():
    settings = DiscussionSettings(random_seed=1)
    payload = api.DiscussionCreateRequest(
        topic="Memory architecture",
        objective="Choose a primary memory architecture for the agent system.",
        constraints="Keep latency low\nPreserve auditability",
        rounds=4,
    )

    objective = api._load_json_objective(payload, settings)

    assert objective.goal == payload.objective
    assert objective.constraints == [
        "Keep latency low",
        "Preserve auditability",
    ]
    assert objective.core_topics[0].title == "MEMORY ARCHITECTURE"


def test_analytics_base_url_is_derived_from_event_endpoint(monkeypatch):
    monkeypatch.delenv("ANALYTICS_API_URL", raising=False)
    settings = DiscussionSettings(
        random_seed=1,
        event_endpoint="http://127.0.0.1:8123/api/v1/discussion-events",
    )

    assert api._analytics_base_url(settings) == "http://127.0.0.1:8123"


def test_random_seed_pool_selects_one_seed_for_a_run(monkeypatch):
    settings = DiscussionSettings(
        random_seed=999,
        random_seeds=[11, 29, 47],
    )
    monkeypatch.setattr(config.secrets, "choice", lambda values: values[1])

    selected = settings.with_selected_random_seed()

    assert selected.random_seed == 29
    assert selected.random_seeds == []
    assert settings.random_seed == 999
    assert settings.random_seeds == [11, 29, 47]


def test_random_seed_pool_loads_from_environment(monkeypatch):
    monkeypatch.setenv("DISCUSSION_RANDOM_SEEDS", "[11, 29, 47]")
    monkeypatch.delenv("DISCUSSION_RANDOM_SEED", raising=False)

    settings = DiscussionSettings(_env_file=None)

    assert settings.random_seeds == [11, 29, 47]


@pytest.mark.parametrize("filename", ["objective.txt", "objective.md"])
def test_upload_is_queued_for_background_execution(monkeypatch, tmp_path, filename):
    settings = DiscussionSettings(
        random_seed=None,
        random_seeds=[1, 7],
        output_dir=tmp_path / "outputs",
        logs_dir=tmp_path / "logs",
    )
    store = FileDiscussionArtifactStore(
        settings.resolve_output_dir(), settings.resolve_logs_dir()
    )
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "_artifact_store", lambda _settings: store)
    monkeypatch.setattr(api, "_require_rag_ready", lambda: None)
    monkeypatch.setattr(config.secrets, "choice", lambda values: values[-1])
    tasks = BackgroundTasks()
    upload = UploadFile(
        filename=filename,
        file=BytesIO(
            b"""OBJECTIVE
Choose a memory architecture.

PRIMARY CONSTRAINTS
- Keep latency low

CORE TOPICS TO DISCUSS
1. Architecture trade-offs

FINAL GOAL
Recommend one architecture.
"""
        ),
    )

    result = api.create_discussion_from_upload(
        _request(),
        tasks,
        upload,
        rounds=6,
        participant_selection_mode="random",
        num_participants=4,
        selected_participant_ids="[]",
    )

    assert result["status"] == "queued"
    assert len(tasks.tasks) == 1
    snapshot = json.loads(
        store.snapshot_path(result["discussion_id"]).read_text(encoding="utf-8")
    )
    assert snapshot["status"] == "queued"
    assert snapshot["configuration"]["num_rounds"] == 6
    assert snapshot["configuration"]["random_seed"] == 7
    assert snapshot["configuration"]["random_seeds"] == []
    assert snapshot["configuration"]["participant_selection_mode"] == "random"
    assert snapshot["configuration"]["selected_participant_ids"] == []
    assert snapshot["progress"]["total_rounds"] == 6


def test_custom_participant_selection_is_validated_and_persisted(monkeypatch, tmp_path):
    selected = [
        "product_user_advocate",
        "practical_architect",
        "safety_governance_advocate",
    ]
    settings = DiscussionSettings(
        random_seed=7,
        output_dir=tmp_path / "outputs",
        logs_dir=tmp_path / "logs",
    )
    store = FileDiscussionArtifactStore(
        settings.resolve_output_dir(), settings.resolve_logs_dir()
    )
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "_artifact_store", lambda _settings: store)
    monkeypatch.setattr(api, "_require_rag_ready", lambda: None)
    tasks = BackgroundTasks()
    payload = api.DiscussionCreateRequest(
        topic="Memory architecture",
        objective="Choose a primary memory architecture for the agent system.",
        rounds=3,
        participant_selection_mode="custom",
        num_participants=3,
        selected_participant_ids=selected,
    )

    result = api.create_discussion(_request(), payload, tasks)

    snapshot = json.loads(
        store.snapshot_path(result["discussion_id"]).read_text(encoding="utf-8")
    )
    assert snapshot["participants"] == selected
    assert snapshot["configuration"]["participant_selection_mode"] == "custom"
    assert snapshot["configuration"]["num_participants"] == 3
    assert snapshot["configuration"]["selected_participant_ids"] == selected


def test_custom_participant_selection_requires_three_agents():
    with pytest.raises(ValueError, match="at least three"):
        DiscussionSettings(
            random_seed=1,
            participant_selection_mode="custom",
            num_participants=3,
            selected_participant_ids=[
                "practical_architect",
                "product_user_advocate",
            ],
        )


def test_discussion_start_is_temporarily_rejected_while_rag_warms(monkeypatch):
    monkeypatch.setattr(
        api,
        "_rag_health_snapshot",
        lambda: (False, "http://127.0.0.1:8000", "connection refused"),
    )

    with pytest.raises(api.HTTPException) as raised:
        api._require_rag_ready()

    assert raised.value.status_code == 503
    assert raised.value.detail["code"] == "rag_warming_up"
    assert raised.value.headers == {"Retry-After": "10"}


def test_api_lifespan_does_not_wait_for_rag_readiness(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        api.week2_adapter,
        "start_local_rag",
        lambda: calls.append("start") or True,
    )
    monkeypatch.setattr(
        api.week2_adapter,
        "stop_local_rag",
        lambda: calls.append("stop"),
    )
    fake_app = SimpleNamespace(state=SimpleNamespace())

    async def exercise() -> None:
        async with api.lifespan(fake_app):
            assert fake_app.state.rag_started_by_api is True
            assert fake_app.state.rag_startup_error is None
            assert calls == ["start"]

    asyncio.run(exercise())

    assert calls == ["start", "stop"]


def test_api_lifespan_survives_rag_start_failure(monkeypatch):
    def fail_to_start() -> bool:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(api.week2_adapter, "start_local_rag", fail_to_start)
    fake_app = SimpleNamespace(state=SimpleNamespace())

    async def exercise() -> None:
        async with api.lifespan(fake_app):
            assert fake_app.state.rag_started_by_api is False
            assert "model unavailable" in fake_app.state.rag_startup_error

    asyncio.run(exercise())


def test_discussion_admission_control_bounds_running_and_queued_work(monkeypatch):
    settings = DiscussionSettings(
        random_seed=1,
        max_concurrent_discussions=1,
        max_queued_discussions=1,
    )
    monkeypatch.setattr(api, "_reserved_discussions", 0)

    assert api._reserve_discussion_capacity(settings) is True
    assert api._reserve_discussion_capacity(settings) is True
    assert api._reserve_discussion_capacity(settings) is False

    api._release_discussion_capacity()
    assert api._reserve_discussion_capacity(settings) is True


def test_history_lists_lightweight_snapshots_newest_first(monkeypatch, tmp_path):
    settings = DiscussionSettings(
        random_seed=1,
        output_dir=tmp_path / "outputs",
        logs_dir=tmp_path / "logs",
    )
    store = FileDiscussionArtifactStore(
        settings.resolve_output_dir(), settings.resolve_logs_dir()
    )
    store.save_snapshot(
        {
            "discussion_id": "discussion-old",
            "status": "failed",
            "objective": {"goal": "Older objective"},
            "participants": ["agent_a"],
            "rounds": [],
            "progress": {"completed_rounds": 0, "total_rounds": 3},
        }
    )
    store.save_snapshot(
        {
            "discussion_id": "discussion-new",
            "status": "completed",
            "objective": {"goal": "Newest objective"},
            "participants": ["agent_a", "agent_b"],
            "rounds": [{"round": 1}],
            "progress": {"completed_rounds": 1, "total_rounds": 1},
            "decision_summary": {
                "preferred_candidate": "candidate_a",
                "consensus_status": "majority",
            },
            "conclusion": "A detailed final recommendation.",
        }
    )
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "_artifact_store", lambda _settings: store)

    result = api.list_discussions(limit=20, offset=0, discussion_status="completed")

    assert result["total"] == 1
    assert result["has_more"] is False
    assert result["items"][0] == {
        "discussion_id": "discussion-new",
        "title": "Newest objective",
        "status": "completed",
        "participants": ["agent_a", "agent_b"],
        "completed_rounds": 1,
        "total_rounds": 1,
        "started_at": None,
        "updated_at": None,
        "completed_at": None,
        "preferred_candidate": "candidate_a",
        "consensus_status": "majority",
        "conclusion_preview": "A detailed final recommendation.",
    }


def test_replay_uses_durable_event_journal(monkeypatch, tmp_path):
    settings = DiscussionSettings(
        random_seed=1,
        output_dir=tmp_path / "outputs",
        logs_dir=tmp_path / "logs",
    )
    store = FileDiscussionArtifactStore(
        settings.resolve_output_dir(), settings.resolve_logs_dir()
    )
    started_snapshot = {
        "discussion_id": "discussion-1",
        "status": "preparing",
        "progress": {"completed_rounds": 0, "total_rounds": 3},
        "rounds": [],
    }
    round_snapshot = {
        **started_snapshot,
        "status": "running",
        "progress": {"completed_rounds": 1, "total_rounds": 3},
        "rounds": [{"round": 1}],
    }
    store.save_snapshot(round_snapshot)
    store.append_event(
        make_discussion_event(
            DiscussionEventType.STARTED,
            started_snapshot,
            sequence=0,
        )
    )
    store.append_event(
        make_discussion_event(
            DiscussionEventType.ROUND_COMPLETED,
            round_snapshot,
            sequence=1,
            round_data={"round": 1},
        )
    )
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "_artifact_store", lambda _settings: store)

    result = api.get_discussion_replay("discussion-1")

    assert [frame["event_type"] for frame in result["frames"]] == [
        "discussion.started",
        "round.completed",
    ]
    assert result["frames"][1]["completed_round"] == 1
    assert result["warnings"] == []


def test_replay_falls_back_for_legacy_snapshot(monkeypatch, tmp_path):
    settings = DiscussionSettings(
        random_seed=1,
        output_dir=tmp_path / "outputs",
        logs_dir=tmp_path / "logs",
    )
    store = FileDiscussionArtifactStore(
        settings.resolve_output_dir(), settings.resolve_logs_dir()
    )
    store.save_snapshot(
        {
            "discussion_id": "legacy-discussion",
            "status": "completed",
            "rounds": [{"round": 1}, {"round": 2}],
            "updated_at": "2026-09-29T00:00:00Z",
            "completed_at": "2026-09-29T00:00:00Z",
        }
    )
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "_artifact_store", lambda _settings: store)

    result = api.get_discussion_replay("legacy-discussion")

    assert [frame["completed_round"] for frame in result["frames"]] == [0, 1, 2, 2]
    assert result["frames"][-1]["event_type"] == "discussion.completed"
    assert "approximate" in result["warnings"][0]
