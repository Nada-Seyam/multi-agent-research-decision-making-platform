"""Helpers for reading line-oriented model output despite light Markdown."""

from __future__ import annotations

import re
from typing import Collection


_MARKER_LINE_RE = re.compile(
    r"^\s*"
    r"(?:#{1,6}\s+)?"  # optional Markdown heading
    r"(?:(?:[-+>]|\d+[.)])\s+)?"  # optional list/quote prefix
    r"(?:\*\*|__)?"  # optional emphasis before the label
    r"(?P<name>[A-Z][A-Z0-9_]*)"
    r"(?:"
    r"(?:\*\*|__)\s*:\s*"  # **LABEL**: value
    r"|"
    r":\s*(?:\*\*|__)?"  # LABEL: value or **LABEL:** value
    r")"
    r"(?P<value>.*?)"
    r"\s*$",
    re.IGNORECASE,
)


def parse_marker_line(
    line: str,
    allowed_names: Collection[str] | None = None,
) -> tuple[str, str] | None:
    """Parse ``LABEL: value`` with common Markdown label formatting.

    Models frequently emit required protocol labels as ``**LABEL:**`` or
    ``**LABEL**:``.  Treat formatting around the label as presentation rather
    than part of the protocol, while leaving Markdown inside the value intact.
    """
    normalized = line.replace(r"\*", "*")
    match = _MARKER_LINE_RE.match(normalized)
    if match is None:
        return None
    name = match.group("name").upper()
    if allowed_names is not None and name not in allowed_names:
        return None
    value = match.group("value").strip()
    return name, value
