"""Cross-family heuristic guidance for proposal selection."""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from sciona.heuristic_registries import resolve_local_heuristic_registry
from sciona.heuristics import HeuristicActionClass
from sciona.principal.expansion_assets import load_local_expansion_assets_by_family
from sciona.principal.heuristic_outcomes import heuristic_action_bonus


class HeuristicProposalGuidance(BaseModel):
    """Shared proposal guidance derived from runtime heuristics and family registries."""

    family: str = ""
    heuristic_ids: list[str] = Field(default_factory=list)
    preferred_action_classes: list[HeuristicActionClass] = Field(default_factory=list)
    registry_asset_id: str = ""
    notes: list[str] = Field(default_factory=list)


def _extract_runtime_heuristic_ids(runtime_artifacts: dict[str, Any]) -> list[str]:
    heuristics = runtime_artifacts.get("heuristics", [])
    if not isinstance(heuristics, list):
        return []
    heuristic_ids: list[str] = []
    for item in heuristics:
        if not isinstance(item, dict):
            continue
        heuristic = item.get("heuristic", {})
        if not isinstance(heuristic, dict):
            continue
        heuristic_id = str(heuristic.get("heuristic_id", "") or "").strip()
        if heuristic_id:
            heuristic_ids.append(heuristic_id)
    return sorted(set(heuristic_ids))


def build_heuristic_proposal_guidance(
    *,
    planning_artifact: dict[str, Any] | None,
    runtime_artifacts: dict[str, Any] | None,
    search_trace: list[dict[str, Any]] | None = None,
) -> HeuristicProposalGuidance:
    """Build deterministic family-local proposal guidance from persisted heuristics."""
    planning_artifact = planning_artifact if isinstance(planning_artifact, dict) else {}
    runtime_artifacts = runtime_artifacts if isinstance(runtime_artifacts, dict) else {}
    family = str(planning_artifact.get("family_hint", "") or "")
    heuristic_ids = _extract_runtime_heuristic_ids(runtime_artifacts)
    if not family or not heuristic_ids:
        return HeuristicProposalGuidance(family=family, heuristic_ids=heuristic_ids)

    registry = resolve_local_heuristic_registry(family)
    if registry is None:
        return HeuristicProposalGuidance(family=family, heuristic_ids=heuristic_ids)

    scores: Counter[HeuristicActionClass] = Counter()
    notes: list[str] = []
    for entry in registry.entries:
        if entry.heuristic_id not in heuristic_ids:
            continue
        priorities = entry.action_priority or entry.supported_action_classes
        for index, action_class in enumerate(priorities):
            scores[action_class] += max(1, 8 - index)
        if entry.escalation_conditions:
            notes.extend(entry.escalation_conditions[:1])

    for action_class, bonus in heuristic_action_bonus(
        family=family,
        heuristic_ids=heuristic_ids,
        search_trace=search_trace,
    ).items():
        scores[action_class] += bonus
        notes.append(f"outcome_memory:{action_class.value}")

    preferred_action_classes = [
        action_class
        for action_class, _score in sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0].value),
        )
    ]
    return HeuristicProposalGuidance(
        family=family,
        heuristic_ids=heuristic_ids,
        preferred_action_classes=preferred_action_classes,
        registry_asset_id=registry.asset_id,
        notes=notes[:4],
    )


def candidate_action_classes(
    candidate_type: str,
    *,
    family: str = "",
    rules_applied: list[str] | None = None,
    applied_assets: list[dict[str, Any]] | None = None,
) -> list[HeuristicActionClass]:
    """Map a candidate proposal into generic action classes."""
    if candidate_type == "local_mutation":
        return [HeuristicActionClass.REPLACE_STAGE]
    if candidate_type == "redecomposition":
        return [
            HeuristicActionClass.SPLIT_STAGE,
            HeuristicActionClass.BRANCH_AND_COMPARE,
        ]

    classes: list[HeuristicActionClass] = []
    for item in applied_assets or []:
        if not isinstance(item, dict):
            continue
        for raw in item.get("action_classes", []) or []:
            try:
                classes.append(HeuristicActionClass(str(raw)))
            except ValueError:
                continue
    if classes:
        return list(dict.fromkeys(classes))

    if family:
        asset = load_local_expansion_assets_by_family().get(family)
        if asset is not None:
            for rule_name in rules_applied or []:
                operation = asset.operation(rule_name)
                if operation is not None:
                    classes.extend(operation.action_classes)
    return list(dict.fromkeys(classes))
