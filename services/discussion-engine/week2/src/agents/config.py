import os
from functools import lru_cache

import yaml

_personas_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "configs",
    "Personas.yaml"
)


@lru_cache(maxsize=1)
def get_personas() -> dict:
    """Load the full Personas.yaml and return the parsed dict."""

    with open(_personas_path, "r") as f:
        return yaml.safe_load(f)


def get_persona(persona_id: str) -> dict:
    """Return a single persona config by id, or raise KeyError."""
    personas = get_personas().get("personas", {})
    if persona_id not in personas:
        available = ", ".join(sorted(personas.keys())) or "<none>"
        raise KeyError(
            f"Unknown persona_id: {persona_id!r}. "
            f"Available personas: {available}"
        )
    persona = personas[persona_id]
    if not isinstance(persona, dict) or persona_id == "shared_rules":
        raise KeyError(f"{persona_id!r} is not a selectable persona.")
    result = dict(persona)
    shared_rules = personas.get("shared_rules")
    if isinstance(shared_rules, dict):
        result["_shared_rules"] = shared_rules
    return result
