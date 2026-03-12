"""Deterministic wrapper for the ingester_hoist_state prompt."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

_ATTRS_RE = re.compile(r"^Cross-window attributes:\s*(.+)$", re.MULTILINE)
_PLAN_RE = re.compile(
    r"Macro-atom plan:\n(?P<plan>.*?)\n\nReturn JSON:",
    re.DOTALL,
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _parse_hoist_prompt(user: str) -> tuple[list[str], list[dict[str, Any]]]:
    attrs_match = _ATTRS_RE.search(user)
    attrs_raw = attrs_match.group(1).strip() if attrs_match else "[]"
    try:
        attrs = ast.literal_eval(attrs_raw)
    except (SyntaxError, ValueError):
        attrs = []
    if not isinstance(attrs, list):
        attrs = []
    normalized_attrs = [str(attr).strip() for attr in attrs if str(attr).strip()]

    plan_match = _PLAN_RE.search(user)
    plan_raw = plan_match.group("plan").strip() if plan_match else "[]"
    try:
        plan = json.loads(plan_raw)
    except json.JSONDecodeError:
        plan = []
    if not isinstance(plan, list):
        plan = []
    return normalized_attrs, [item for item in plan if isinstance(item, dict)]


def _pascal_case(text: str) -> str:
    parts = [part for part in _TOKEN_RE.findall(text) if part]
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _model_name(macro_plan: list[dict[str, Any]]) -> str:
    if len(macro_plan) == 1:
        base = _pascal_case(str(macro_plan[0].get("name", "") or "Pipeline"))
        return f"{base}State"
    return "PipelineState"


def _infer_attr_type(attr: str, macro_plan: list[dict[str, Any]]) -> str:
    for atom in macro_plan:
        for io in list(atom.get("inputs", []) or []) + list(atom.get("outputs", []) or []):
            if not isinstance(io, dict):
                continue
            if str(io.get("name", "") or "").strip() == attr:
                type_desc = str(io.get("type_desc", "") or "").strip()
                if type_desc:
                    return type_desc
    return "Any"


def _hoist_from_attrs(
    cross_window_attrs: list[str],
    macro_plan: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not cross_window_attrs:
        return None

    fields: list[list[str]] = []
    unknown_count = 0
    for attr in cross_window_attrs:
        inferred = _infer_attr_type(attr, macro_plan)
        if inferred == "Any":
            unknown_count += 1
        fields.append([attr, inferred])

    if unknown_count > len(cross_window_attrs) / 2:
        return None

    state_model = {
        "model_name": _model_name(macro_plan),
        "fields": fields,
        "source_attrs": list(cross_window_attrs),
        "docstring": (
            f"Hoisted cross-window state for {len(cross_window_attrs)} persistent attributes."
        ),
    }
    return {"state_models": [state_model]}


class ASTStateHoister:
    """Deterministic ingester state hoister with LLM fallback."""

    _telemetry_provider = "deterministic"
    _telemetry_model = "state_hoister_v1"

    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback
        self._last_completion_metadata: dict[str, Any] = {}
        self._last_error_metadata: dict[str, Any] = {}

    def get_last_completion_metadata(self) -> dict[str, Any]:
        return dict(self._last_completion_metadata)

    def get_last_error_metadata(self) -> dict[str, Any]:
        return dict(self._last_error_metadata)

    async def complete(self, system: str, user: str) -> str:
        attrs, macro_plan = _parse_hoist_prompt(user)
        result = _hoist_from_attrs(attrs, macro_plan)
        if result is not None:
            self._last_completion_metadata = {
                "state_hoist_source": "deterministic",
                "state_model_count": len(result.get("state_models", [])),
            }
            self._last_error_metadata = {}
            return json.dumps(result)

        self._last_completion_metadata = {"state_hoist_source": "fallback"}
        self._last_error_metadata = {}
        return await self._fallback.complete(system, user)

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)
