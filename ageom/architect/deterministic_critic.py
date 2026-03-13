"""Deterministic wrapper for architect_critique approvals."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ageom.architect.models import ConceptType, IOSpec


@dataclass(frozen=True)
class CritiqueNode:
    name: str
    concept_type: str
    inputs: list[IOSpec]
    outputs: list[IOSpec]
    status: str
    matched_primitive: str


@dataclass(frozen=True)
class CritiquePrompt:
    parent_name: str
    parent_description: str
    parent_inputs: list[IOSpec]
    parent_outputs: list[IOSpec]
    sub_nodes: list[CritiqueNode]


_SUB_NODE_RE = re.compile(
    r"^\s*-\s*(?P<name>.*?)\s+\[(?P<concept>[^\]]+)\]\s+"
    r"\(inputs:\s*(?P<inputs>.*?),\s*outputs:\s*(?P<outputs>.*?),\s*"
    r"status:\s*(?P<status>[^,]+),\s*matched_primitive:\s*(?P<primitive>.*?)\)\s*$",
    re.MULTILINE,
)
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _extract_prompt_value(user: str, label: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(label)}:\s*(.*)$", re.MULTILINE)
    match = pattern.search(user)
    return match.group(1).strip() if match is not None else ""


def _parse_ios(raw: str) -> list[IOSpec]:
    text = raw.strip()
    if not text or text.lower() == "none":
        return []
    parts = re.split(r", (?=[A-Za-z_][A-Za-z0-9_]*: )", text)
    ios: list[IOSpec] = []
    for part in parts:
        if ":" not in part:
            continue
        name, rest = part.split(":", 1)
        required = "(optional" not in rest
        default_match = re.search(r"default=([^)]+)", rest)
        type_desc = re.sub(r"\s*\(optional.*\)$", "", rest).strip()
        ios.append(
            IOSpec(
                name=name.strip(),
                type_desc=type_desc or "Any",
                required=required,
                default_value_repr=default_match.group(1).strip() if default_match else "",
            )
        )
    return ios


def _parse_critique_prompt(user: str) -> CritiquePrompt:
    sub_nodes: list[CritiqueNode] = []
    sub_nodes_block = user.split("Proposed sub-nodes:\n", 1)
    if len(sub_nodes_block) == 2:
        block = sub_nodes_block[1].split("\n\nProposed edges:", 1)[0]
        for match in _SUB_NODE_RE.finditer(block):
            sub_nodes.append(
                CritiqueNode(
                    name=match.group("name").strip(),
                    concept_type=match.group("concept").strip(),
                    inputs=_parse_ios(match.group("inputs")),
                    outputs=_parse_ios(match.group("outputs")),
                    status=match.group("status").strip(),
                    matched_primitive=match.group("primitive").strip(),
                )
            )
    return CritiquePrompt(
        parent_name=_extract_prompt_value(user, "Name"),
        parent_description=_extract_prompt_value(user, "Description"),
        parent_inputs=_parse_ios(_extract_prompt_value(user, "Inputs")),
        parent_outputs=_parse_ios(_extract_prompt_value(user, "Outputs")),
        sub_nodes=sub_nodes,
    )


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower().replace("-", "_")))


def _concept_baseline(child_concept: str, parent_tokens: set[str]) -> float:
    try:
        concept = ConceptType(child_concept)
    except ValueError:
        return 0.0
    concept_tokens = _tokenize(concept.value.replace("_", " "))
    overlap = len(concept_tokens & parent_tokens)
    if not concept_tokens:
        return 0.0
    return overlap / len(concept_tokens)


def _check_structural_issues(prompt: CritiquePrompt) -> list[str]:
    """Lightweight structural checks on the parsed critique prompt."""
    issues: list[str] = []

    # Duplicate child names
    seen_names: set[str] = set()
    for child in prompt.sub_nodes:
        lower_name = child.name.lower().strip()
        if lower_name in seen_names:
            issues.append(f"Duplicate sub-node name: '{child.name}'")
        seen_names.add(lower_name)

    # Near-duplicate child names (Jaccard ≥ 0.85)
    child_token_sets = [(_tokenize(c.name), c.name) for c in prompt.sub_nodes]
    for i, (left_tokens, left_name) in enumerate(child_token_sets):
        if not left_tokens:
            continue
        for right_tokens, right_name in child_token_sets[i + 1:]:
            if not right_tokens:
                continue
            union = len(left_tokens | right_tokens)
            similarity = len(left_tokens & right_tokens) / union if union else 0.0
            if similarity >= 0.85:
                issues.append(
                    f"Near-duplicate sub-nodes: '{left_name}' and '{right_name}'"
                )

    # Children with no inputs AND no outputs are suspicious
    for child in prompt.sub_nodes:
        if not child.inputs and not child.outputs:
            issues.append(f"Sub-node '{child.name}' has no inputs and no outputs")

    return issues


def _semantic_critique(prompt: CritiquePrompt) -> tuple[bool, str]:
    if len(prompt.sub_nodes) < 2:
        return False, "need at least two sub-nodes for deterministic approval"

    # Structural validation before semantic checks
    structural_issues = _check_structural_issues(prompt)
    if structural_issues:
        return False, f"structural issues: {'; '.join(structural_issues)}"

    parent_input_names = {io.name for io in prompt.parent_inputs}
    parent_output_names = {io.name for io in prompt.parent_outputs}
    child_input_names = {io.name for node in prompt.sub_nodes for io in node.inputs}
    child_output_names = {io.name for node in prompt.sub_nodes for io in node.outputs}

    input_coverage = (
        len(parent_input_names & child_input_names) / len(parent_input_names)
        if parent_input_names
        else 1.0
    )
    output_coverage = (
        len(parent_output_names & child_output_names) / len(parent_output_names)
        if parent_output_names
        else 1.0
    )

    parent_tokens = _tokenize(f"{prompt.parent_name} {prompt.parent_description}")
    parent_name_tokens = _tokenize(prompt.parent_name)
    min_relevance = 1.0
    max_triviality = 0.0
    for child in prompt.sub_nodes:
        primitive_tokens = set()
        if child.matched_primitive and child.matched_primitive != "(none)":
            primitive_tokens = _tokenize(child.matched_primitive)
        child_name_tokens = _tokenize(child.name)
        child_tokens = child_name_tokens | primitive_tokens
        overlap_ratio = (
            len(child_tokens & parent_tokens) / len(child_tokens)
            if child_tokens
            else 0.0
        )
        relevance = max(overlap_ratio, _concept_baseline(child.concept_type, parent_tokens))
        min_relevance = min(min_relevance, relevance)

        union = len(parent_name_tokens | child_name_tokens)
        triviality = len(parent_name_tokens & child_name_tokens) / union if union else 0.0
        max_triviality = max(max_triviality, triviality)

    if input_coverage < 1.0:
        return False, "parent inputs not fully covered by child inputs"
    if output_coverage < 0.8:
        return False, "parent outputs not sufficiently covered by child outputs"
    if min_relevance < 0.3:
        return False, "child nodes are not semantically aligned enough for deterministic approval"
    if max_triviality > 0.85:
        return False, "child decomposition is too close to the parent task"

    return True, "Deterministic semantic critique passed"


class DeterministicCritic:
    """Deterministic architect critique gate with LLM fallback."""

    _telemetry_provider = "deterministic"
    _telemetry_model = "critic_v1"

    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback
        self._last_completion_metadata: dict[str, Any] = {}
        self._last_error_metadata: dict[str, Any] = {}

    def get_last_completion_metadata(self) -> dict[str, Any]:
        return dict(self._last_completion_metadata)

    def get_last_error_metadata(self) -> dict[str, Any]:
        return dict(self._last_error_metadata)

    async def complete(self, system: str, user: str) -> str:
        prompt = _parse_critique_prompt(user)
        approved, reason = _semantic_critique(prompt)
        if approved:
            self._last_completion_metadata = {
                "critique_source": "deterministic",
                "critique_sub_node_count": len(prompt.sub_nodes),
            }
            self._last_error_metadata = {}
            return json.dumps(
                {
                    "approved": True,
                    "reason": reason,
                    "io_issues": [],
                    "flagged_nodes": [],
                }
            )

        self._last_completion_metadata = {
            "critique_source": "fallback",
            "critique_gate_reason": reason,
        }
        self._last_error_metadata = {}
        return await self._fallback.complete(system, user)

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)
