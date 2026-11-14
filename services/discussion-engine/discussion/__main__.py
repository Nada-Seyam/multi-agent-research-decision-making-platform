"""CLI for the Week 3 discussion runner."""
from __future__ import annotations

import argparse
import sys

from .artifacts import FileDiscussionArtifactStore, NullDiscussionArtifactStore
from .config import get_settings
from .delivery import replay_pending_events
from .event_sinks import build_event_sink
from .runner import run_discussion
from .week2_adapter import load_objective


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a discussion.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--objective",
        help="Path to a Week 2 objective file.",
    )
    action.add_argument(
        "--replay-events",
        metavar="DISCUSSION_ID",
        help="Replay locally journaled events that the consumer has not acknowledged.",
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    store = (
        FileDiscussionArtifactStore(
            settings.resolve_output_dir(),
            settings.resolve_logs_dir(),
        )
        if settings.persist_artifacts
        else NullDiscussionArtifactStore()
    )
    sink = build_event_sink(settings)

    if args.replay_events:
        if not settings.persist_artifacts:
            print("event replay requires DISCUSSION_PERSIST_ARTIFACTS=true", file=sys.stderr)
            return 2
        if not settings.event_endpoint:
            print("event replay requires DISCUSSION_EVENT_ENDPOINT", file=sys.stderr)
            return 2
        try:
            assert isinstance(store, FileDiscussionArtifactStore)
            summary = replay_pending_events(args.replay_events, store, sink)
        except Exception as exc:
            print(f"event replay failed: {exc}", file=sys.stderr)
            return 1
        print(
            f"event replay finished: attempted={summary.attempted}, "
            f"delivered={summary.delivered}, pending={summary.pending}"
        )
        return 0 if summary.pending == 0 else 1

    try:
        objective = load_objective(args.objective, settings.resolve_week2_root())
        result = run_discussion(
            objective,
            settings=settings,
            event_sink=sink,
            artifact_store=store,
        )
    except Exception as exc:
        print(f"discussion failed: {exc}", file=sys.stderr)
        return 1

    result_file = store.snapshot_path(result.discussion_id)
    if result_file is None:
        print("discussion completed without local artifact persistence")
    else:
        print(f"discussion result saved to {result_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
