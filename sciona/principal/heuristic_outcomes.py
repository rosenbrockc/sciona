"""Heuristic outcome memory extracted from persisted search traces."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from pydantic import BaseModel, Field

from sciona.heuristics import HeuristicActionClass


class HeuristicOutcomeRecord(BaseModel):
    """One persisted heuristic/action outcome from a proposal-selection trial."""

    family: str = ""
    heuristic_ids: list[str] = Field(default_factory=list)
    proposal_label: str = ""
    candidate_action_classes: list[str] = Field(default_factory=list)
    selected: bool = False
    loss_delta: float = 0.0


class HeuristicUsabilityScopeMemory(BaseModel):
    """Compact scope-level usability memory for one evaluated trial."""

    scope: str
    usable: bool
    confidence: float = 0.0
    blocking_reason_codes: list[str] = Field(default_factory=list)
    warning_reason_codes: list[str] = Field(default_factory=list)
    provenance_kinds: list[str] = Field(default_factory=list)


class HeuristicUsabilityMemoryRecord(BaseModel):
    """Long-term usability memory for one heuristic-signature outcome."""

    trial: int = 0
    family: str = ""
    proposal_label: str = ""
    heuristic_signature: list[str] = Field(default_factory=list)
    heuristic_ids: list[str] = Field(default_factory=list)
    candidate_action_classes: list[str] = Field(default_factory=list)
    selected: bool = False
    loss_delta: float = 0.0
    context: dict[str, Any] = Field(default_factory=dict)
    usability_assessment_id: str = ""
    usable_for_guidance: bool = False
    usable_for_scoring: bool = False
    usable_for_final_benchmark: bool = False
    usability_scopes: dict[str, HeuristicUsabilityScopeMemory] = Field(
        default_factory=dict
    )
    heuristic_summary: dict[str, Any] = Field(default_factory=dict)


def _compact_heuristic_cohort_member(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    heuristics: list[dict[str, Any]] = []
    for item in raw.get("heuristics", []) or []:
        if not isinstance(item, dict):
            continue
        heuristic = item.get("heuristic", {})
        if not isinstance(heuristic, dict):
            continue
        heuristic_id = str(heuristic.get("heuristic_id", "") or "").strip()
        if not heuristic_id:
            continue
        heuristics.append(
            {
                "heuristic_id": heuristic_id,
                "confidence": float(
                    item.get("confidence", heuristic.get("confidence", 0.0)) or 0.0
                ),
                "source_section": str(item.get("source_section", "") or ""),
            }
        )
    usability = raw.get("usability", {})
    usability_summary = {}
    if isinstance(usability, dict):
        scope_exclusions = usability.get("scope_exclusions", {})
        usability_summary = {
            "usable_for_guidance": bool(usability.get("usable_for_guidance", False)),
            "usable_for_scoring": bool(usability.get("usable_for_scoring", False)),
            "usable_for_final_benchmark": bool(
                usability.get("usable_for_final_benchmark", False)
            ),
            "scope_exclusions": (
                dict(scope_exclusions) if isinstance(scope_exclusions, dict) else {}
            ),
        }
    return {
        "member_label": str(raw.get("member_label", "") or ""),
        "tracker_value": str(raw.get("tracker_value", "") or ""),
        "loss": raw.get("loss"),
        "heuristics": heuristics,
        "usability": usability_summary,
    }


def _compact_heuristic_cohort(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in (
        "cohort_size",
        "evaluated_member_count",
        "attempted_member_count",
        "source_tracker_path",
        "combined_tracker_csv",
    ):
        value = raw.get(key)
        if value is not None:
            summary[key] = value
    heuristics = raw.get("heuristics")
    if isinstance(heuristics, dict):
        summary["heuristics"] = dict(heuristics)
    gating_heuristics = raw.get("gating_heuristics")
    if isinstance(gating_heuristics, dict):
        summary["gating_heuristics"] = dict(gating_heuristics)
    usability = raw.get("usability")
    if isinstance(usability, dict):
        summary["usability"] = dict(usability)
    excluded_members = raw.get("excluded_members", [])
    if isinstance(excluded_members, list):
        compact_members = [
            compact
            for compact in (
                _compact_heuristic_cohort_member(item) for item in excluded_members
            )
            if compact
        ]
        if compact_members:
            summary["excluded_members"] = compact_members[:20]
    return summary


def summarize_runtime_heuristic_evidence(
    runtime_artifacts: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep only the compact runtime-evidence fields needed for long-term memory."""
    if not isinstance(runtime_artifacts, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in (
        "runtime_context",
        "telemetry_summary",
        "heuristics",
        "heuristic_summary",
        "usability_assessment",
    ):
        value = runtime_artifacts.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            summary[key] = value
    heuristic_cohort = _compact_heuristic_cohort(runtime_artifacts.get("heuristic_cohort"))
    if heuristic_cohort:
        summary["heuristic_cohort"] = heuristic_cohort
    return summary


def _scope_memory_record(raw: Any) -> HeuristicUsabilityScopeMemory | None:
    if not isinstance(raw, dict):
        return None
    provenance_kinds: list[str] = []
    for item in raw.get("provenance", []) or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "") or "").strip()
        if kind:
            provenance_kinds.append(kind)
    blocking_reason_codes: list[str] = []
    for item in raw.get("blocking_reasons", []) or []:
        if isinstance(item, dict):
            code = str(item.get("code", "") or "").strip()
            if code:
                blocking_reason_codes.append(code)
    warning_reason_codes: list[str] = []
    for item in raw.get("warning_reasons", []) or []:
        if isinstance(item, dict):
            code = str(item.get("code", "") or "").strip()
            if code:
                warning_reason_codes.append(code)
    return HeuristicUsabilityScopeMemory(
        scope=str(raw.get("scope", "") or "").strip(),
        usable=bool(raw.get("usable", False)),
        confidence=float(raw.get("confidence", 0.0) or 0.0),
        blocking_reason_codes=sorted(dict.fromkeys(blocking_reason_codes)),
        warning_reason_codes=sorted(dict.fromkeys(warning_reason_codes)),
        provenance_kinds=sorted(dict.fromkeys(provenance_kinds)),
    )


def _compact_usability_assessment(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    scopes: dict[str, Any] = {}
    for scope_name in ("guidance", "scoring", "final_benchmark"):
        scope = _scope_memory_record(raw.get(scope_name))
        if scope is not None:
            scopes[scope_name] = scope.model_dump(mode="json")
    return {
        "assessment_id": str(raw.get("assessment_id", "") or ""),
        "family": str(raw.get("family", "") or ""),
        "task_intent": str(raw.get("task_intent", "") or ""),
        "heuristic_signature": [
            str(item)
            for item in raw.get("heuristic_signature", []) or []
            if str(item)
        ],
        "required_contracts_checked": [
            str(item)
            for item in raw.get("required_contracts_checked", []) or []
            if str(item)
        ],
        "usable_for_guidance": bool(raw.get("usable_for_guidance", False)),
        "usable_for_scoring": bool(raw.get("usable_for_scoring", False)),
        "usable_for_final_benchmark": bool(
            raw.get("usable_for_final_benchmark", False)
        ),
        "confidence": float(raw.get("confidence", 0.0) or 0.0),
        "uncertainty_notes": [
            str(item)
            for item in raw.get("uncertainty_notes", []) or []
            if str(item)
        ],
        "guidance": scopes.get("guidance", {}),
        "scoring": scopes.get("scoring", {}),
        "final_benchmark": scopes.get("final_benchmark", {}),
    }


def _compact_context(entry: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for key in ("trial", "thread_id", "loss"):
        if key in entry:
            context[key] = entry.get(key)
    for key in ("planning_artifact", "structure", "admissibility", "expansion", "rollback"):
        value = entry.get(key)
        if isinstance(value, dict):
            context[key] = dict(value)
    runtime_evidence = summarize_runtime_heuristic_evidence(
        entry.get("runtime_evidence", entry.get("runtime_artifacts", {}))
    )
    if runtime_evidence:
        context["runtime_evidence"] = runtime_evidence
    return context


def extract_heuristic_usability_memory(
    trial_history: list[dict[str, Any]],
) -> list[HeuristicUsabilityMemoryRecord]:
    """Extract deterministic long-term usability memory from trial history."""
    records: list[HeuristicUsabilityMemoryRecord] = []
    for entry in trial_history:
        if not isinstance(entry, dict):
            continue
        proposal = entry.get("proposal_selection", {})
        if not isinstance(proposal, dict):
            continue
        baseline_loss = proposal.get("baseline_loss")
        try:
            baseline_loss_value = float(baseline_loss)
        except (TypeError, ValueError):
            continue
        selected_label = str(proposal.get("selected", "") or "")
        runtime_evidence = summarize_runtime_heuristic_evidence(
            entry.get("runtime_evidence", entry.get("runtime_artifacts", {}))
        )
        usability_assessment = _compact_usability_assessment(
            runtime_evidence.get("usability_assessment", {})
        )
        heuristic_signature = [
            str(item)
            for item in usability_assessment.get("heuristic_signature", []) or []
            if str(item)
        ]
        heuristic_summary = dict(runtime_evidence.get("heuristic_summary", {}))
        context = _compact_context(entry)
        if runtime_evidence:
            context["runtime_evidence"] = runtime_evidence
        family = str(
            entry.get("family")
            or entry.get("planning_artifact", {}).get("family_hint", "")
            or ""
        )
        for candidate in proposal.get("candidates", []) or []:
            if not isinstance(candidate, dict):
                continue
            try:
                candidate_loss = float(candidate.get("loss"))
            except (TypeError, ValueError):
                continue
            evidence = candidate.get("evidence", {})
            if not isinstance(evidence, dict):
                evidence = {}
            heuristic_ids = [
                str(item)
                for item in evidence.get("heuristic_ids", []) or []
                if str(item)
            ]
            action_classes = [
                str(item)
                for item in evidence.get("candidate_action_classes", []) or []
                if str(item)
            ]
            if not heuristic_ids or not action_classes:
                continue
            scope_memory: dict[str, HeuristicUsabilityScopeMemory] = {}
            for scope_name in ("guidance", "scoring", "final_benchmark"):
                scope = usability_assessment.get(scope_name, {})
                if isinstance(scope, dict):
                    scope_record = _scope_memory_record(scope)
                    if scope_record is not None:
                        scope_memory[scope_name] = scope_record
            records.append(
                HeuristicUsabilityMemoryRecord(
                    trial=int(entry.get("trial", 0) or 0),
                    family=family,
                    proposal_label=str(candidate.get("label", "") or ""),
                    heuristic_signature=heuristic_signature or list(heuristic_ids),
                    heuristic_ids=heuristic_ids,
                    candidate_action_classes=action_classes,
                    selected=str(candidate.get("label", "") or "") == selected_label,
                    loss_delta=baseline_loss_value - candidate_loss,
                    context=context,
                    usability_assessment_id=str(
                        usability_assessment.get("assessment_id", "") or ""
                    ),
                    usable_for_guidance=bool(
                        usability_assessment.get("usable_for_guidance", False)
                    ),
                    usable_for_scoring=bool(
                        usability_assessment.get("usable_for_scoring", False)
                    ),
                    usable_for_final_benchmark=bool(
                        usability_assessment.get("usable_for_final_benchmark", False)
                    ),
                    usability_scopes=scope_memory,
                    heuristic_summary=heuristic_summary,
                )
            )
    return records


def summarize_heuristic_usability_memory(
    records: list[HeuristicUsabilityMemoryRecord],
) -> dict[str, Any]:
    """Summarize long-term usability memory for persistence and reporting."""
    positive = [record for record in records if record.loss_delta > 0.0]
    selected = [record for record in records if record.selected]
    signature_stats: dict[str, dict[str, Any]] = {}
    action_counts: Counter[str] = Counter()
    scope_counts = {
        "guidance": 0,
        "scoring": 0,
        "final_benchmark": 0,
    }
    for record in records:
        signature_key = "|".join(record.heuristic_signature or record.heuristic_ids)
        stats = signature_stats.setdefault(
            signature_key,
            {
                "record_count": 0,
                "selected_count": 0,
                "positive_count": 0,
                "total_loss_delta": 0.0,
                "max_loss_delta": 0.0,
                "heuristic_ids": sorted(dict.fromkeys(record.heuristic_ids)),
                "action_classes": set(),
                "family": record.family,
                "context_keys": sorted(record.context.keys()),
            },
        )
        stats["record_count"] += 1
        stats["total_loss_delta"] += record.loss_delta
        stats["max_loss_delta"] = max(float(stats["max_loss_delta"]), record.loss_delta)
        if record.selected:
            stats["selected_count"] += 1
        if record.loss_delta > 0.0:
            stats["positive_count"] += 1
        stats["action_classes"].update(record.candidate_action_classes)
        action_counts.update(record.candidate_action_classes)
        if record.usable_for_guidance:
            scope_counts["guidance"] += 1
        if record.usable_for_scoring:
            scope_counts["scoring"] += 1
        if record.usable_for_final_benchmark:
            scope_counts["final_benchmark"] += 1
    summarized_signatures: dict[str, Any] = {}
    for signature_key in sorted(signature_stats):
        stats = signature_stats[signature_key]
        record_count = int(stats["record_count"])
        summarized_signatures[signature_key] = {
            "record_count": record_count,
            "selected_count": int(stats["selected_count"]),
            "positive_count": int(stats["positive_count"]),
            "mean_loss_delta": (
                float(stats["total_loss_delta"] / record_count)
                if record_count > 0
                else 0.0
            ),
            "max_loss_delta": float(stats["max_loss_delta"]),
            "heuristic_ids": list(stats["heuristic_ids"]),
            "action_classes": sorted(stats["action_classes"]),
            "family": stats["family"],
            "context_keys": list(stats["context_keys"]),
        }
    return {
        "memory_count": len(records),
        "selected_memory_count": len(selected),
        "positive_memory_count": len(positive),
        "mean_positive_loss_delta": (
            float(sum(record.loss_delta for record in positive) / len(positive))
            if positive
            else 0.0
        ),
        "scope_support_counts": scope_counts,
        "mature_action_classes": sorted(
            action_class
            for action_class, count in action_counts.items()
            if count >= 2
        ),
        "heuristic_signatures": summarized_signatures,
        "records": [record.model_dump(mode="json") for record in records],
    }


def extract_heuristic_outcomes(search_trace: list[dict[str, Any]]) -> list[HeuristicOutcomeRecord]:
    """Extract heuristic/action/loss records from persisted proposal traces."""
    records: list[HeuristicOutcomeRecord] = []
    for entry in search_trace:
        if not isinstance(entry, dict):
            continue
        proposal = entry.get("proposal_selection", {})
        if not isinstance(proposal, dict):
            continue
        baseline_loss = proposal.get("baseline_loss")
        try:
            baseline_loss_value = float(baseline_loss)
        except (TypeError, ValueError):
            continue
        selected_label = str(proposal.get("selected", "") or "")
        for candidate in proposal.get("candidates", []) or []:
            if not isinstance(candidate, dict):
                continue
            try:
                candidate_loss = float(candidate.get("loss"))
            except (TypeError, ValueError):
                continue
            evidence = candidate.get("evidence", {})
            if not isinstance(evidence, dict):
                evidence = {}
            heuristic_ids = [
                str(item)
                for item in evidence.get("heuristic_ids", []) or []
                if str(item)
            ]
            action_classes = [
                str(item)
                for item in evidence.get("candidate_action_classes", []) or []
                if str(item)
            ]
            if not heuristic_ids or not action_classes:
                continue
            records.append(
                HeuristicOutcomeRecord(
                    family=str(candidate.get("family", "") or ""),
                    heuristic_ids=heuristic_ids,
                    proposal_label=str(candidate.get("label", "") or ""),
                    candidate_action_classes=action_classes,
                    selected=str(candidate.get("label", "") or "") == selected_label,
                    loss_delta=baseline_loss_value - candidate_loss,
                )
            )
    return records


def summarize_heuristic_outcomes(records: list[HeuristicOutcomeRecord]) -> dict[str, Any]:
    """Summarize persisted heuristic outcomes for benchmark/reporting surfaces."""
    positive = [record for record in records if record.loss_delta > 0.0]
    selected = [record for record in records if record.selected]
    action_counts: Counter[str] = Counter()
    for record in positive:
        action_counts.update(record.candidate_action_classes)
    return {
        "outcome_count": len(records),
        "positive_outcome_count": len(positive),
        "selected_outcome_count": len(selected),
        "mean_positive_loss_delta": (
            float(sum(record.loss_delta for record in positive) / len(positive))
            if positive
            else 0.0
        ),
        "mature_action_classes": sorted(
            action_class
            for action_class, count in action_counts.items()
            if count >= 2
        ),
    }


def heuristic_action_bonus(
    *,
    family: str,
    heuristic_ids: list[str],
    search_trace: list[dict[str, Any]] | None,
) -> Counter[HeuristicActionClass]:
    """Return a cautious same-run action prior from repeated positive outcomes."""
    bonuses: Counter[HeuristicActionClass] = Counter()
    if not family or not heuristic_ids or not isinstance(search_trace, list):
        return bonuses

    gains_by_action: dict[HeuristicActionClass, list[float]] = defaultdict(list)
    heuristic_set = set(heuristic_ids)
    for record in extract_heuristic_outcomes(search_trace):
        if record.family != family:
            continue
        if not heuristic_set.intersection(record.heuristic_ids):
            continue
        if record.loss_delta <= 0.0:
            continue
        for raw in record.candidate_action_classes:
            try:
                gains_by_action[HeuristicActionClass(raw)].append(record.loss_delta)
            except ValueError:
                continue

    for action_class, gains in gains_by_action.items():
        if len(gains) < 2:
            continue
        mean_gain = sum(gains) / len(gains)
        if mean_gain > 0.0:
            bonuses[action_class] += 1
    return bonuses
