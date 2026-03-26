"""Benchmark data models and small scoring helpers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from sciona.architect.models import ConceptType


@dataclass(frozen=True)
class FlowLeafSpec:
    name: str
    description: str
    type_signature: str
    query_hint: str
    declaration_name: str
    inputs: tuple[tuple[str, str], ...]
    outputs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class FlowBenchmarkCase:
    case_id: str
    domain: str
    prompt: str
    concept_type: ConceptType
    leaves: tuple[FlowLeafSpec, ...]


@dataclass
class FlowBenchmarkResult:
    case_id: str
    domain: str
    variant: str
    execution_path: str
    ok: bool
    latency_ms: float
    prompt_calls: int
    matched_leaves: int
    total_leaves: int
    node_count: int
    leaf_coverage: float = 0.0
    best_similarity: float = 0.0
    decomposition_depth: int = 0
    decomposition_leaf_count: int = 0
    decomposition_edge_count: int = 0
    planner_tool_dispatches: int = 0
    planner_tool_latency_ms: float = 0.0
    planner_escalation_count: int = 0
    planner_termination_reason: str = ""
    planner_action_signature: str = ""
    planner_actions: tuple[str, ...] = ()
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlowBenchmarkAggregate:
    variant: str
    execution_paths: list[str] = field(default_factory=list)
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    avg_latency_ms: float = 0.0
    total_prompt_calls: int = 0
    avg_prompt_calls: float = 0.0
    total_planner_tool_dispatches: int = 0
    avg_planner_tool_dispatches: float = 0.0
    total_planner_tool_latency_ms: float = 0.0
    avg_planner_tool_latency_ms: float = 0.0
    total_planner_escalations: int = 0
    avg_planner_escalations: float = 0.0
    planner_termination_counts: dict[str, int] = field(default_factory=dict)
    planner_action_counts: dict[str, int] = field(default_factory=dict)
    planner_action_signature_counts: dict[str, int] = field(default_factory=dict)
    dominant_planner_termination_reason: str = ""
    dominant_planner_action_signature: str = ""
    avg_leaf_coverage: float = 0.0
    avg_best_similarity: float = 0.0
    repeat_groups: int = 0
    stable_groups: int = 0
    stability_rate: float = 1.0
    _case_outcomes: dict[str, list[bool]] = field(default_factory=dict)

    def record(self, result: FlowBenchmarkResult) -> None:
        self.total_cases += 1
        if result.execution_path not in self.execution_paths:
            self.execution_paths.append(result.execution_path)
        if result.ok:
            self.passed_cases += 1
        else:
            self.failed_cases += 1

        total_latency = self.avg_latency_ms * (self.total_cases - 1)
        total_latency += result.latency_ms
        self.avg_latency_ms = total_latency / max(1, self.total_cases)

        self.total_prompt_calls += int(result.prompt_calls)
        self.avg_prompt_calls = self.total_prompt_calls / max(1, self.total_cases)

        self.total_planner_tool_dispatches += int(result.planner_tool_dispatches)
        self.avg_planner_tool_dispatches = (
            self.total_planner_tool_dispatches / max(1, self.total_cases)
        )

        self.total_planner_tool_latency_ms += float(result.planner_tool_latency_ms)
        self.avg_planner_tool_latency_ms = (
            self.total_planner_tool_latency_ms / max(1, self.total_cases)
        )

        self.total_planner_escalations += int(result.planner_escalation_count)
        self.avg_planner_escalations = self.total_planner_escalations / max(
            1, self.total_cases
        )

        if result.planner_termination_reason:
            self.planner_termination_counts[result.planner_termination_reason] = (
                int(self.planner_termination_counts.get(result.planner_termination_reason, 0) or 0)
                + 1
            )
        if result.planner_action_signature:
            self.planner_action_signature_counts[result.planner_action_signature] = (
                int(
                    self.planner_action_signature_counts.get(
                        result.planner_action_signature, 0
                    )
                    or 0
                )
                + 1
            )
        for action in result.planner_actions:
            self.planner_action_counts[action] = (
                int(self.planner_action_counts.get(action, 0) or 0) + 1
            )

        prev = self.total_cases - 1
        self.avg_leaf_coverage = (
            self.avg_leaf_coverage * prev + result.leaf_coverage
        ) / self.total_cases
        self.avg_best_similarity = (
            self.avg_best_similarity * prev + result.best_similarity
        ) / self.total_cases
        self._case_outcomes.setdefault(result.case_id, []).append(result.ok)

    def finalize(self) -> None:
        groups = list(self._case_outcomes.values())
        self.repeat_groups = len(groups)
        self.stable_groups = sum(1 for outcomes in groups if len(set(outcomes)) <= 1)
        self.stability_rate = (
            self.stable_groups / self.repeat_groups if self.repeat_groups else 1.0
        )
        self.dominant_planner_termination_reason = _dominant_count_key(
            self.planner_termination_counts
        )
        self.dominant_planner_action_signature = _dominant_count_key(
            self.planner_action_signature_counts
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "execution_paths": sorted(self.execution_paths),
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "avg_latency_ms": self.avg_latency_ms,
            "total_prompt_calls": self.total_prompt_calls,
            "avg_prompt_calls": self.avg_prompt_calls,
            "total_planner_tool_dispatches": self.total_planner_tool_dispatches,
            "avg_planner_tool_dispatches": self.avg_planner_tool_dispatches,
            "total_planner_tool_latency_ms": self.total_planner_tool_latency_ms,
            "avg_planner_tool_latency_ms": self.avg_planner_tool_latency_ms,
            "total_planner_escalations": self.total_planner_escalations,
            "avg_planner_escalations": self.avg_planner_escalations,
            "planner_termination_counts": dict(
                sorted(self.planner_termination_counts.items())
            ),
            "planner_action_counts": dict(sorted(self.planner_action_counts.items())),
            "planner_action_signature_counts": dict(
                sorted(self.planner_action_signature_counts.items())
            ),
            "dominant_planner_termination_reason": self.dominant_planner_termination_reason,
            "dominant_planner_action_signature": self.dominant_planner_action_signature,
            "avg_leaf_coverage": self.avg_leaf_coverage,
            "avg_best_similarity": self.avg_best_similarity,
            "repeat_groups": self.repeat_groups,
            "stable_groups": self.stable_groups,
            "stability_rate": self.stability_rate,
        }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _dominant_count_key(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    return min(counts.items(), key=lambda item: (-int(item[1] or 0), str(item[0])))[0]


def _hint_matches(text: str, hint: str) -> bool:
    text_tokens = {token for token in _slug(text).split("_") if token}
    hint_tokens = {token for token in _slug(hint).split("_") if token}
    if not text_tokens or not hint_tokens:
        return False
    return hint_tokens <= text_tokens or len(text_tokens & hint_tokens) >= min(
        2, len(hint_tokens)
    )
