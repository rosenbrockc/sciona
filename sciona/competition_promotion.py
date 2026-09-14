"""Lossless competition-template intake and implementation preflight.

This is an intake check, not a publication grant. A resolved dependency does
not prove stage semantics, whole-graph execution, or review completion.
"""

from collections import Counter
from dataclasses import dataclass
import json
import inspect
from pathlib import Path
from typing import Any, Mapping

from sciona.architect.skeleton_assets import SkeletonFamilyAsset


@dataclass(frozen=True)
class CompetitionCandidate:
    original_template: dict[str, Any]
    original_bindings: dict[str, Any]
    asset: SkeletonFamilyAsset


def assess_keyword_call_contract(input_names: list[str], implementation: Any) -> dict[str, Any]:
    """Check the runner's keyword-call contract without executing a dependency.

    This establishes argument compatibility only, not type/shape correctness or
    semantic equivalence. Defaults and **kwargs are honored; positional-only
    parameters require an explicit adapter in the keyword-based graph runner.
    """
    signature = inspect.signature(implementation)
    parameters = signature.parameters
    supplied = set(input_names)
    required = {
        name for name, p in parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }
    keyword_names = {
        name for name, p in parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    positional = {
        name for name, p in parameters.items()
        if p.kind == inspect.Parameter.POSITIONAL_ONLY
        and (name in supplied or p.default is inspect.Parameter.empty)
    }
    accepts_extra = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    missing = sorted(required - supplied)
    unexpected = sorted(supplied - keyword_names) if not accepts_extra else []
    return {
        "keyword_compatible": not (missing or unexpected or positional or len(supplied) != len(input_names)),
        "missing_required_parameters": missing,
        "unexpected_parameters": unexpected,
        "positional_adapter_required": sorted(positional),
        "duplicate_inputs": len(supplied) != len(input_names),
    }


def load_competition_candidates(directory: Path) -> list[CompetitionCandidate]:
    """Validate publication shape while retaining every source field.

    Never round-trip through the narrower skeleton model to save a template:
    applicability, boundary contracts, and binding evidence must be retained.
    """
    candidates = []
    identities = set()
    for path in sorted(directory.glob("*.json")):
        if path.stem.endswith("_bindings"):
            continue
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or "stages" not in raw:
            continue
        asset = SkeletonFamilyAsset.model_validate(raw)
        if asset.asset_id in identities:
            raise ValueError("duplicate competition template identity")
        identities.add(asset.asset_id)
        binding_path = path.with_name(path.stem + "_bindings.json")
        bindings = json.loads(binding_path.read_text()) if binding_path.exists() else {"bindings": []}
        if not isinstance(bindings, dict) or not isinstance(bindings.get("bindings"), list):
            raise ValueError("invalid competition binding report")
        candidates.append(CompetitionCandidate(raw, bindings, asset))
    return candidates


def assess_implementation_intake(
    candidate: CompetitionCandidate,
    catalog: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Count exact, active, currently served stage targets without exemptions.

    Orchestration and external stages still need executable implementations.
    Historical labels and approximate substitutions do not satisfy this check.
    """
    stages = [stage.stage_id for stage in candidate.asset.stages]
    stage_set = set(stages)
    bindings: dict[str, list[dict[str, Any]]] = {}
    blockers: Counter[str] = Counter()
    if not stages:
        blockers["empty_graph"] += 1
    if len(stage_set) != len(stages):
        blockers["duplicate_stage"] += len(stages) - len(stage_set)
    for row in candidate.original_bindings["bindings"]:
        if not isinstance(row, dict):
            blockers["invalid_binding"] += 1
            continue
        stage_id = row.get("stage_id")
        if stage_id not in stage_set:
            blockers["orphan_binding"] += 1
            continue
        bindings.setdefault(stage_id, []).append(row)
    resolved = 0
    pinned = 0
    for stage_id in stages:
        rows = bindings.get(stage_id, [])
        if len(rows) != 1:
            blockers["missing_binding" if not rows else "ambiguous_binding"] += 1
            continue
        row = rows[0]
        if row.get("status") != "active":
            blockers["approximate_binding" if row.get("status") == "approximate" else "unresolved_binding"] += 1
            continue
        target = row.get("bound_artifact_fqdn")
        if not target:
            blockers["missing_implementation"] += 1
            continue
        artifact = catalog.get(target)
        if artifact is None:
            blockers["target_not_in_catalog"] += 1
            continue
        if artifact.get("status") != "approved" or artifact.get("is_publishable") is not True:
            blockers["target_not_served"] += 1
            continue
        resolved += 1
        version = row.get("bound_version_content_hash")
        if not version:
            blockers["dependency_version_unpinned"] += 1
        elif version != artifact.get("content_hash"):
            blockers["dependency_version_not_current"] += 1
        else:
            pinned += 1
    return {
        "stage_count": len(stages),
        "active_served_stage_count": resolved,
        "current_pinned_stage_count": pinned,
        "implementation_coverage": resolved / len(stages) if stages else 0.0,
        "dependency_intake_complete": not blockers,
        "blockers": dict(sorted(blockers.items())),
        # Deliberately no 'approved' or 'publishable' field. Replay, semantic
        # review, provenance, and project-specific gates remain separate.
    }


def summarize_competition_intake(
    candidates: list[CompetitionCandidate],
    catalog: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    reports = [assess_implementation_intake(candidate, catalog) for candidate in candidates]
    blockers: Counter[str] = Counter()
    for report in reports:
        blockers.update(report["blockers"])
    return {
        "templates": len(reports),
        "stages": sum(r["stage_count"] for r in reports),
        "templates_with_active_served_implementation": sum(r["active_served_stage_count"] > 0 for r in reports),
        "templates_with_complete_dependency_intake": sum(r["dependency_intake_complete"] for r in reports),
        "active_served_stages": sum(r["active_served_stage_count"] for r in reports),
        "current_pinned_stages": sum(r["current_pinned_stage_count"] for r in reports),
        "stage_blockers": dict(sorted(blockers.items())),
    }
