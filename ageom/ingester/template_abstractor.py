"""Deterministic wrapper for the ingester_abstract prompt."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SECTION_RE = re.compile(r"^(Atom|Description|Concept type|Inputs|Outputs|Source methods):\s*(.*)$")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# --- In-code defaults (used when config file is missing) ---
_DEFAULT_DOMAIN_PREFIXES = ("ecg", "audio", "financial", "bio", "image", "video")
_DEFAULT_ABSTRACT_NAME_OVERRIDES: dict[str, str] = {
    "signal_filter": "Signal Conditioner",
    "signal_transform": "Representation Transformer",
    "graph_optimization": "Graph Path Optimizer",
    "dynamic_programming": "State Recurrence Solver",
}
_DEFAULT_TRANSFORM_TEMPLATES: dict[str, str] = {
    "signal_filter": "Applies a constrained filtering step that attenuates or preserves components of a structured signal while keeping the signal representation stable.",
    "signal_transform": "Maps a structured input into an alternative representation that exposes frequency, basis, or latent structure for downstream computation.",
    "graph_optimization": "Computes an optimized path or score over a connected structure by propagating local costs into a global selection.",
    "dynamic_programming": "Builds an output by reusing intermediate state across overlapping subproblems in a staged recurrence.",
}
_DEFAULT_APPLICATIONS: dict[str, list[str]] = {
    "signal_filter": ["industrial sensing", "medical monitoring", "geophysics"],
    "signal_transform": ["communications", "robotics", "scientific imaging"],
    "graph_optimization": ["transport logistics", "network routing", "supply planning"],
    "dynamic_programming": ["computational biology", "resource planning", "control systems"],
}
_DEFAULT_PROPERTIES: dict[str, list[str]] = {
    "signal_filter": ["deterministic", "signal_processing", "shape_preserving_candidate"],
    "signal_transform": ["deterministic", "representation_change"],
    "graph_optimization": ["deterministic", "cost_sensitive"],
    "dynamic_programming": ["deterministic", "stateful_recurrence"],
}


def _load_abstractor_config(
    path: str = "",
) -> tuple[tuple[str, ...], dict[str, str], dict[str, str], dict[str, list[str]], dict[str, list[str]]]:
    """Load abstractor config from JSON, falling back to in-code defaults."""
    if not path:
        default = Path(__file__).parent / ".." / "data" / "abstractor_config.json"
        if default.exists():
            path = str(default)
    if path:
        try:
            with open(path) as f:
                data = json.load(f)
            return (
                tuple(data.get("domain_prefixes", _DEFAULT_DOMAIN_PREFIXES)),
                data.get("abstract_name_overrides", _DEFAULT_ABSTRACT_NAME_OVERRIDES),
                data.get("transform_templates", _DEFAULT_TRANSFORM_TEMPLATES),
                data.get("applications", _DEFAULT_APPLICATIONS),
                data.get("properties", _DEFAULT_PROPERTIES),
            )
        except Exception:
            logger.debug("Failed to load abstractor config from %s", path)
    return (
        _DEFAULT_DOMAIN_PREFIXES,
        _DEFAULT_ABSTRACT_NAME_OVERRIDES,
        _DEFAULT_TRANSFORM_TEMPLATES,
        _DEFAULT_APPLICATIONS,
        _DEFAULT_PROPERTIES,
    )


_DOMAIN_PREFIXES, _ABSTRACT_NAME_OVERRIDES, _TRANSFORM_TEMPLATES, _APPLICATIONS, _PROPERTIES = (
    _load_abstractor_config()
)


def _parse_abstract_prompt(user: str) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "atom_name": "",
        "atom_description": "",
        "concept_type": "",
        "inputs": [],
        "outputs": [],
        "method_names": [],
    }
    current = ""
    for raw_line in user.splitlines():
        line = raw_line.rstrip()
        match = _SECTION_RE.match(line)
        if match:
            current = match.group(1).lower()
            rest = match.group(2).strip()
            if current == "atom":
                fields["atom_name"] = rest
            elif current == "description":
                fields["atom_description"] = rest
            elif current == "concept type":
                fields["concept_type"] = rest
            elif current == "source methods":
                fields["method_names"] = [part.strip() for part in rest.split(",") if part.strip()]
            continue
        if current in {"inputs", "outputs"} and line.strip():
            fields[current].append(line.strip().lstrip("- ").strip())
    return fields


def _normalized_tokens(text: str) -> list[str]:
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok]


def _abstract_name(atom_name: str, concept_type: str) -> str:
    base = atom_name.strip()
    tokens = [tok for tok in _normalized_tokens(base) if tok not in _DOMAIN_PREFIXES]
    if concept_type in _ABSTRACT_NAME_OVERRIDES:
        return _ABSTRACT_NAME_OVERRIDES[concept_type]
    if len(tokens) < 2:
        return ""
    return " ".join(token.capitalize() for token in tokens[:3])


def _shape_property(inputs: list[str], outputs: list[str]) -> list[str]:
    if not inputs or not outputs:
        return []
    if any(inp.split(":", 1)[-1].strip() == out.split(":", 1)[-1].strip() for inp in inputs for out in outputs):
        return ["shape_preserving_candidate"]
    return []


def _generate_abstract(
    atom_name: str,
    concept_type: str,
    inputs: list[str],
    outputs: list[str],
    methods: list[str],
) -> dict[str, Any] | None:
    if len(_normalized_tokens(atom_name)) < 2 or not concept_type or not inputs or not outputs:
        return None
    abstract_name = _abstract_name(atom_name, concept_type)
    if not abstract_name:
        return None
    transform = _TRANSFORM_TEMPLATES.get(
        concept_type,
        f"Transforms {len(inputs)} typed inputs into {len(outputs)} outputs through {max(1, len(methods))} structured processing steps.",
    )
    properties = list(dict.fromkeys(_PROPERTIES.get(concept_type, ["deterministic"]) + _shape_property(inputs, outputs)))
    applications = _APPLICATIONS.get(
        concept_type,
        ["scientific computing", "automation", "data engineering"],
    )
    return {
        "abstract_name": abstract_name,
        "conceptual_transform": transform,
        "abstract_inputs": inputs,
        "abstract_outputs": outputs,
        "algorithmic_properties": properties,
        "cross_disciplinary_applications": applications[:3],
    }


class TemplateAbstractor:
    """Deterministic conceptual-profile generator with LLM fallback."""

    _telemetry_provider = "deterministic"
    _telemetry_model = "template_abstractor_v1"

    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback
        self._last_completion_metadata: dict[str, Any] = {}
        self._last_error_metadata: dict[str, Any] = {}

    def get_last_completion_metadata(self) -> dict[str, Any]:
        return dict(self._last_completion_metadata)

    def get_last_error_metadata(self) -> dict[str, Any]:
        return dict(self._last_error_metadata)

    async def complete(self, system: str, user: str) -> str:
        parsed = _parse_abstract_prompt(user)
        result = _generate_abstract(
            parsed["atom_name"],
            parsed["concept_type"],
            parsed["inputs"],
            parsed["outputs"],
            parsed["method_names"],
        )
        if result is not None:
            self._last_completion_metadata = {
                "abstract_source": "deterministic",
                "concept_type": parsed["concept_type"],
            }
            self._last_error_metadata = {}
            return json.dumps(result)

        self._last_completion_metadata = {"abstract_source": "fallback"}
        self._last_error_metadata = {}
        return await self._fallback.complete(system, user)

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)
