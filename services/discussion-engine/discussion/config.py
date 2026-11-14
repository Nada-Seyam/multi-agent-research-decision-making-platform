"""Configuration for the Week 3 discussion runner.

Secrets and model credentials remain owned by Week 2's configuration. These
settings only control the Week 3 discussion boundary.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DiscussionSettings(BaseSettings):
    """Validated, environment-backed discussion settings."""

    num_participants: int = Field(default=4, ge=3, le=6)
    participant_selection_mode: Literal["random", "custom"] = "random"
    selected_participant_ids: list[str] = Field(default_factory=list)
    num_rounds: int = Field(default=4, ge=3, le=6)
    random_seed: int | None = None
    random_seeds: list[int] = Field(default_factory=list)
    week2_root: Path = Path("week2")
    output_dir: Path = Path("outputs/discussions")
    logs_dir: Path = Path("logs/discussions")
    max_initial_queries: int = Field(default=4, ge=1)
    max_evidence_chunks_per_query: int = Field(default=4, ge=1)
    max_evidence_chars_per_chunk: int = Field(default=6000, ge=500)
    max_total_evidence_chunks: int = Field(default=50, ge=1)
    max_persona_evidence_chars: int = Field(default=24_000, ge=4_000)
    max_parallel_agents: int = Field(default=4, ge=1)
    max_concurrent_discussions: int = Field(default=2, ge=1, le=32)
    max_queued_discussions: int = Field(default=8, ge=1, le=256)
    dynamic_challenger_routing: bool = True
    utility_consistency_tolerance: float = Field(default=0.25, ge=0.0)
    persist_artifacts: bool = True
    event_endpoint: str | None = None
    event_token: SecretStr | None = Field(default=None, exclude=True)
    event_timeout_seconds: float = Field(default=10.0, gt=0.0)
    event_max_attempts: int = Field(default=5, ge=1)
    event_retry_initial_seconds: float = Field(default=1.0, ge=0.0)
    event_retry_max_seconds: float = Field(default=30.0, ge=0.0)
    event_delivery_required: bool = False

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="DISCUSSION_",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_random_seed_source(self) -> Self:
        """Require either the legacy seed or a pool used for per-run selection."""
        if self.random_seed is None and not self.random_seeds:
            raise ValueError(
                "Set DISCUSSION_RANDOM_SEEDS to a JSON list of integers or "
                "DISCUSSION_RANDOM_SEED to one integer."
            )
        selected = self.selected_participant_ids
        if self.participant_selection_mode == "random":
            if selected:
                raise ValueError(
                    "selected_participant_ids must be empty when participant selection is random"
                )
        else:
            if len(selected) < 3:
                raise ValueError("A custom discussion requires at least three participants.")
            if len(selected) != len(set(selected)):
                raise ValueError("Custom participant ids must be unique.")
            if self.num_participants != len(selected):
                raise ValueError(
                    "num_participants must match the number of custom participant ids"
                )
        return self

    def with_selected_random_seed(self) -> Self:
        """Return run-scoped settings with one seed selected from the pool."""
        if not self.random_seeds:
            return self
        return self.model_copy(
            update={
                "random_seed": secrets.choice(self.random_seeds),
                "random_seeds": [],
            }
        )

    def resolve_week2_root(self) -> Path:
        """Return the Week 2 root as an absolute project-local path."""
        return self._resolve_project_path(self.week2_root)

    def resolve_output_dir(self) -> Path:
        """Return the configured output directory as an absolute path."""
        return self._resolve_project_path(self.output_dir)

    def resolve_logs_dir(self) -> Path:
        """Return the configured heavy-data log directory as an absolute path."""
        return self._resolve_project_path(self.logs_dir)

    @staticmethod
    def _resolve_project_path(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path


def get_settings() -> DiscussionSettings:
    """Load the current settings from the Week 3 environment."""
    return DiscussionSettings()
