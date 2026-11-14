"""Coordinate the high-level lifecycle of a multi-agent discussion."""

from __future__ import annotations

from typing import Any

from . import week2_adapter
from .artifacts import DiscussionArtifactStore
from .conclusion import ConclusionQualityWarning, generate_conclusion
from .config import DiscussionSettings, get_settings
from .decision import synthesize_decision
from .event_sinks import DiscussionEventSink, EventDeliveryError, build_event_sink
from .events import DiscussionEventType, utc_now
from .graph import build_week2_agent_graph
from .identifiers import discussion_id
from .lifecycle import (
    DiscussionLifecycle,
    build_result_snapshot,
    default_artifact_store,
)
from .models import DiscussionResult, DiscussionRound, OpinionResult
from .participants import select_participant_ids
from .research import build_shared_research, dossier_for_prompt
from .round_execution import commit_round, execute_round


class DiscussionOrchestrator:
    """Coordinate setup, round barriers, and terminal discussion states."""

    def __init__(
        self,
        objective: Any,
        settings: DiscussionSettings | None = None,
        *,
        event_sink: DiscussionEventSink | None = None,
        artifact_store: DiscussionArtifactStore | None = None,
        run_id: str | None = None,
    ):
        self.objective = objective
        self.settings = (settings or get_settings()).with_selected_random_seed()
        self._owns_event_sink = event_sink is None
        sink = event_sink or build_event_sink(self.settings)
        store = artifact_store or default_artifact_store(self.settings)
        self.lifecycle = DiscussionLifecycle(self.settings, sink, store)
        self.run_id = run_id

    def run(self) -> DiscussionResult:
        settings = self.settings
        personas = week2_adapter.load_week2_personas(settings.resolve_week2_root())
        participants = select_participant_ids(
            personas,
            num_participants=settings.num_participants,
            random_seed=settings.random_seed,
            selected_participant_ids=(
                settings.selected_participant_ids
                if settings.participant_selection_mode == "custom"
                else None
            ),
        )
        graph = build_week2_agent_graph(
            participants,
            random_seed=settings.random_seed,
        )
        graph_data = graph.to_dict()
        graph_data["directed"] = True
        run_id = self.run_id or discussion_id(self.objective, participants, settings)
        proposed_alternatives: dict[str, str] = {}
        started_at = utc_now()
        research = None
        history: dict[str, list[OpinionResult]] = {
            persona_id: [] for persona_id in participants
        }
        rounds: list[DiscussionRound] = []
        message_sequence = 0
        stage = "starting"
        conclusion = None
        warnings: list[dict[str, Any]] = []
        preparation: dict[str, Any] = {
            "stage": "room",
            "message": "Agent room created",
            "completed_units": 0,
            "total_units": settings.max_initial_queries,
            "queries": [],
        }
        result = build_result_snapshot(
            discussion_id=run_id,
            objective=self.objective,
            participants=participants,
            graph=graph_data,
            research=None,
            rounds=rounds,
            settings=settings,
            proposed_alternatives=proposed_alternatives,
            status="preparing",
            started_at=started_at,
            preparation=preparation,
        )

        try:
            self.lifecycle.publish(
                result,
                DiscussionEventType.STARTED,
                sequence=0,
            )
            stage = "research"

            def publish_preparation(progress: dict[str, Any]) -> None:
                nonlocal preparation, stage
                preparation = progress
                stage = f"research_{progress.get('stage', 'preparing')}"
                checkpoint = build_result_snapshot(
                    discussion_id=run_id,
                    objective=self.objective,
                    participants=participants,
                    graph=graph_data,
                    research=None,
                    rounds=rounds,
                    settings=settings,
                    proposed_alternatives=proposed_alternatives,
                    status="preparing",
                    started_at=started_at,
                    preparation=preparation,
                )
                self.lifecycle.checkpoint(checkpoint)

            research = build_shared_research(
                self.objective,
                settings=settings,
                progress_callback=publish_preparation,
            )
            prompt_dossier = dossier_for_prompt(
                research.dossier,
                settings.max_persona_evidence_chars,
            )

            for round_number in range(1, settings.num_rounds + 1):
                stage = f"round_{round_number}"
                execution = execute_round(
                    round_number=round_number,
                    objective=self.objective,
                    settings=settings,
                    participants=participants,
                    personas=personas,
                    graph=graph,
                    dossier=prompt_dossier,
                    history=history,
                    proposed_alternatives=proposed_alternatives,
                    message_sequence=message_sequence,
                )
                message_sequence = execution.next_message_sequence
                completed_round = execution.discussion_round
                commit_round(completed_round, history, proposed_alternatives)
                rounds.append(completed_round)

                result = build_result_snapshot(
                    discussion_id=run_id,
                    objective=self.objective,
                    participants=participants,
                    graph=graph_data,
                    research=research,
                    rounds=rounds,
                    settings=settings,
                    proposed_alternatives=proposed_alternatives,
                    status="running",
                    started_at=started_at,
                    preparation=preparation,
                )
                self.lifecycle.publish(
                    result,
                    DiscussionEventType.ROUND_COMPLETED,
                    sequence=round_number,
                    round_data=completed_round.to_dict(),
                )

            stage = "conclusion"
            try:
                conclusion = generate_conclusion(rounds[-1].opinions)
            except ConclusionQualityWarning as warning:
                conclusion = warning.fallback_conclusion
                warnings.append(
                    {
                        "stage": "conclusion",
                        "code": "conclusion_quality",
                        "message": str(warning),
                        "details": list(warning.reasons),
                    }
                )
            stage = "decision_summary"
            result = build_result_snapshot(
                discussion_id=run_id,
                objective=self.objective,
                participants=participants,
                graph=graph_data,
                research=research,
                rounds=rounds,
                settings=settings,
                proposed_alternatives=proposed_alternatives,
                status="completed",
                started_at=started_at,
                decision_summary=synthesize_decision(rounds),
                conclusion=conclusion,
                warnings=warnings,
                completed_at=utc_now(),
                preparation=preparation,
            )
            self.lifecycle.publish(
                result,
                DiscussionEventType.COMPLETED,
                sequence=settings.num_rounds + 1,
            )
            return result
        except Exception as exc:
            self._publish_failure(
                run_id=run_id,
                participants=participants,
                graph=graph_data,
                research=research,
                rounds=rounds,
                proposed_alternatives=proposed_alternatives,
                started_at=started_at,
                stage=stage,
                conclusion=conclusion,
                warnings=warnings,
                preparation=preparation,
                error=exc,
            )
            raise
        finally:
            if self._owns_event_sink:
                self.lifecycle.event_sink.close()

    def _publish_failure(
        self,
        *,
        run_id: str,
        participants: list[str],
        graph: dict[str, Any],
        research: Any,
        rounds: list[DiscussionRound],
        proposed_alternatives: dict[str, str],
        started_at: str,
        stage: str,
        conclusion: str | None,
        warnings: list[dict[str, Any]],
        preparation: dict[str, Any] | None,
        error: Exception,
    ) -> None:
        failure = {
            "stage": stage,
            "round": (
                int(stage.removeprefix("round_"))
                if stage.startswith("round_")
                else None
            ),
            "error_type": type(error).__name__,
            "message": str(error),
            "retryable": isinstance(error, EventDeliveryError),
        }
        failed_result = build_result_snapshot(
            discussion_id=run_id,
            objective=self.objective,
            participants=participants,
            graph=graph,
            research=research,
            rounds=rounds,
            settings=self.settings,
            proposed_alternatives=proposed_alternatives,
            status="failed",
            started_at=started_at,
            failure=failure,
            conclusion=conclusion,
            warnings=warnings,
            preparation=preparation,
        )
        try:
            self.lifecycle.publish(
                failed_result,
                DiscussionEventType.FAILED,
                sequence=self.settings.num_rounds + 2,
                require_delivery=False,
            )
        except Exception:
            pass


def run_discussion(
    objective: Any,
    *,
    settings: DiscussionSettings | None = None,
    event_sink: DiscussionEventSink | None = None,
    artifact_store: DiscussionArtifactStore | None = None,
    run_id: str | None = None,
) -> DiscussionResult:
    """Construct the orchestrator and run one complete discussion."""
    return DiscussionOrchestrator(
        objective,
        settings=settings,
        event_sink=event_sink,
        artifact_store=artifact_store,
        run_id=run_id,
    ).run()
