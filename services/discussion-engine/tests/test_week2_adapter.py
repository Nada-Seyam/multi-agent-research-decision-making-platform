from __future__ import annotations

from pathlib import Path

import pytest

from discussion.week2_adapter import Week2IntegrationError, resolve_week2_root


def test_week2_root_is_resolved_relative_to_week3_project():
    root = resolve_week2_root("week2")

    assert root.name == "week2"
    assert (root / "src" / "agents").is_dir()


def test_missing_week2_root_has_actionable_error(tmp_path: Path):
    with pytest.raises(Week2IntegrationError, match="DISCUSSION_WEEK2_ROOT"):
        resolve_week2_root(tmp_path / "missing-week2")
