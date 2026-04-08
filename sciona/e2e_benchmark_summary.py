"""Helpers for Phase 7 e2e benchmark policy summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sciona.principal.search_policy import (
    evaluate_asset_migration_readiness,
    evaluate_enriched_cdg_policy,
    enforce_anti_shortcut_policy,
    evaluate_behavioral_benchmark_policy,
    summarize_proposal_selection,
    summarize_search_discipline,
    summarize_asset_migration_readiness,
    validate_required_benchmark_artifacts,
)


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        with path.open() as handle:
            return json.load(handle)
    except Exception:
        return None


def _read_matches(label_dir: Path) -> tuple[int, int, list[str]]:
    payload = _read_json(label_dir / "matches.json")
    if not isinstance(payload, list):
        return 0, 0, []
    verified = 0
    names: list[str] = []
    for match in payload:
        if not isinstance(match, dict):
            continue
        verified_match = match.get("verified_match")
        if isinstance(verified_match, dict) and verified_match.get("verified"):
            verified += 1
        candidates = []
        if isinstance(verified_match, dict):
            candidate = verified_match.get("candidate")
            if isinstance(candidate, dict):
                candidates.append(candidate)
        all_candidates = match.get("all_candidates", [])
        if isinstance(all_candidates, list):
            candidates.extend(candidate for candidate in all_candidates if isinstance(candidate, dict))
        for candidate in candidates:
            declaration = candidate.get("declaration")
            if not isinstance(declaration, dict):
                continue
            name = str(declaration.get("name", "")).strip()
            if name:
                names.append(name)
    return len(payload), verified, sorted(set(names))


def _read_planning_artifact(label_dir: Path) -> tuple[dict[str, Any], str]:
    payload = _read_json(label_dir / "cdg.json")
    if not isinstance(payload, dict):
        return {}, ""
    planning = payload.get("planning_artifact")
    if not isinstance(planning, dict):
        metadata = payload.get("metadata", {})
        if isinstance(metadata, dict):
            planning = metadata.get("planning_artifact")
    if not isinstance(planning, dict):
        planning = {}
    family = str(
        planning.get("family_hint")
        or planning.get("paradigm")
        or ""
    ).strip()
    return planning, family


def _read_skeleton_asset(label_dir: Path) -> dict[str, Any]:
    payload = _read_json(label_dir / "cdg.json")
    if not isinstance(payload, dict):
        return {}
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
        asset = metadata.get("skeleton_asset")
        if isinstance(asset, dict):
            return dict(asset)
    planning = payload.get("planning_artifact", {})
    if isinstance(planning, dict):
        skeleton_intent = planning.get("skeleton_intent", {})
        if isinstance(skeleton_intent, dict):
            asset = skeleton_intent.get("asset")
            if isinstance(asset, dict):
                return dict(asset)
    return {}


def _read_applied_assets(trial_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for entry in trial_history:
        if not isinstance(entry, dict):
            continue
        expansion = entry.get("expansion", {})
        if not isinstance(expansion, dict):
            continue
        applied = expansion.get("applied_assets", [])
        if not isinstance(applied, list):
            continue
        for asset in applied:
            if isinstance(asset, dict):
                assets.append(dict(asset))
    return assets


def _artifact_presence(label_dir: Path) -> dict[str, bool]:
    return {
        "planning_artifact": (label_dir / "cdg.json").exists(),
        "final_candidate": (label_dir / "cdg.json").exists(),
        "runtime_context": (
            (label_dir / "runtime_evidence.json").exists()
            or (label_dir / "profile_runtime_artifacts.json").exists()
        ),
        "trial_history": (label_dir / "trial_history.json").exists(),
        "matches": (label_dir / "matches.json").exists(),
    }


def _used_real_assets(matched_primitives: list[str]) -> bool:
    return any(
        name.startswith(("ageoa.", "ageom.", "sciona.expansion_atoms."))
        for name in matched_primitives
    )


def _is_executable(label_dir: Path, *, postprocess: dict[str, Any] | None) -> bool:
    if isinstance(postprocess, dict):
        synth = postprocess.get("synthesize", {})
        export = postprocess.get("export", {})
        profile = postprocess.get("profile", {})
        if isinstance(synth, dict) and isinstance(export, dict) and isinstance(profile, dict):
            return bool(synth.get("compiled_ok")) and int(export.get("exit_code", 1)) == 0 and int(
                profile.get("exit_code", 1)
            ) == 0
    return (label_dir / "cdg.json").exists() and (label_dir / "matches.json").exists()


def _report_to_dict(report: Any) -> dict[str, Any]:
    return {
        "passed": bool(getattr(report, "passed", False)),
        "violations": list(getattr(report, "violations", ()) or ()),
        "warnings": list(getattr(report, "warnings", ()) or ()),
        "details": dict(getattr(report, "details", {}) or {}),
    }


def build_e2e_benchmark_summary(
    *,
    output_dir: Path,
    goal: str,
    prover: str,
    llm_provider: str,
    total_gt: int,
    latencies_ms: dict[str, int],
    ground_truth_hits: dict[str, int],
    variant_dirs: dict[str, Path],
    profile_dataset: str = "",
    shortcut_flags: dict[str, bool] | None = None,
    declared_shortcuts: list[str] | None = None,
) -> dict[str, Any]:
    """Build the persisted e2e benchmark summary with policy metadata."""
    results: dict[str, Any] = {}
    policy_variants: dict[str, Any] = {}
    postprocess_summary: dict[str, Any] = {}

    for variant, label_dir in variant_dirs.items():
        label_dir = Path(label_dir)
        matches_total, matches_verified, matched_primitives = _read_matches(label_dir)
        planning_artifact, family = _read_planning_artifact(label_dir)
        artifact_presence = _artifact_presence(label_dir)
        postprocess = _read_json(label_dir / "postprocess.json")
        trial_history = _read_json(label_dir / "trial_history.json")
        if not isinstance(trial_history, list):
            trial_history = []
        summary = summarize_search_discipline(trial_history)
        proposal_summary = summarize_proposal_selection(trial_history)
        search_summary = {
            "trial_count": summary.trial_count,
            "expansion_attempts": summary.expansion_attempts,
            "admissibility_decisions": summary.admissibility_decisions,
            "pruned_trials": summary.pruned_trials,
            "reused_cached_evaluations": summary.reused_cached_evaluations,
        }
        proposal_selection = {
            "trial_count": proposal_summary.trial_count,
            "proposal_selection_trials": proposal_summary.proposal_selection_trials,
            "selected_trials": proposal_summary.selected_trials,
            "rejected_trials": proposal_summary.rejected_trials,
            "skipped_due_to_admissibility_trials": proposal_summary.skipped_due_to_admissibility_trials,
            "selected_proposal_counts": dict(proposal_summary.selected_proposal_counts),
            "proposal_selection_labels": list(proposal_summary.proposal_selection_labels),
            "mean_selected_proposal_improvement": proposal_summary.mean_selected_proposal_improvement,
            "best_selected_proposal_improvement": proposal_summary.best_selected_proposal_improvement,
        }
        skeleton_asset = _read_skeleton_asset(label_dir)
        applied_assets = _read_applied_assets(trial_history)
        asset_inventory = [asset for asset in [skeleton_asset, *applied_assets] if asset]
        enriched_cdg_policy = evaluate_enriched_cdg_policy(
            {
                "search_discipline": search_summary,
                "proposal_selection": proposal_selection,
                "search_trace_summary": {
                    "entry_count": len(trial_history),
                    "applied_asset_count": len(asset_inventory),
                },
            }
        )
        asset_migration_policy = evaluate_asset_migration_readiness(asset_inventory)
        asset_migration_summary = summarize_asset_migration_readiness(asset_inventory)
        coverage = round(float(ground_truth_hits.get(variant, 0)) / max(total_gt, 1), 2)
        used_real_assets = _used_real_assets(matched_primitives)
        executable = _is_executable(label_dir, postprocess=postprocess if isinstance(postprocess, dict) else None)
        results[variant] = {
            "latency_ms": int(latencies_ms.get(variant, 0) or 0),
            "matches_total": matches_total,
            "matches_verified": matches_verified,
            "ground_truth_hits": int(ground_truth_hits.get(variant, 0) or 0),
            "ground_truth_coverage": coverage,
            "family": family,
            "used_real_assets": used_real_assets,
            "executable": executable,
            "artifact_presence": artifact_presence,
            "matched_primitives": matched_primitives,
            "planning_artifact_present": bool(planning_artifact),
            "asset_inventory": {
                "asset_count": len(asset_inventory),
                "ready_asset_count": asset_migration_summary["ready_asset_count"],
                "blocked_asset_count": asset_migration_summary["blocked_asset_count"],
                "ready_asset_ids": list(asset_migration_summary["ready_asset_ids"]),
                "blocked_asset_ids": list(asset_migration_summary["blocked_asset_ids"]),
            },
            "enriched_cdg": _report_to_dict(enriched_cdg_policy),
            "asset_migration": _report_to_dict(asset_migration_policy),
        }
        results[variant]["search_discipline"] = search_summary
        results[variant]["proposal_selection"] = proposal_selection
        if isinstance(postprocess, dict):
            postprocess_summary[variant] = postprocess

        if variant == "raw_llm":
            continue

        available_artifacts = {
            key: True for key, present in artifact_presence.items() if present
        }
        artifact_policy = validate_required_benchmark_artifacts(
            available_artifacts,
            required_keys=("planning_artifact", "final_candidate", "runtime_context"),
        )
        behavioral_policy = evaluate_behavioral_benchmark_policy(
            results[variant],
            allowed_families={family} if family else {""},
            require_real_assets=True,
        )
        policy_variants[variant] = {
            "required_artifacts": _report_to_dict(artifact_policy),
            "behavioral": _report_to_dict(behavioral_policy),
            "enriched_cdg": _report_to_dict(enriched_cdg_policy),
            "asset_migration": _report_to_dict(asset_migration_policy),
        }

    anti_shortcut = enforce_anti_shortcut_policy(
        {
            "shortcut_flags": dict(shortcut_flags or {}),
            "declared_shortcuts": list(declared_shortcuts or []),
        }
    )

    report: dict[str, Any] = {
        "goal": goal,
        "prover": prover,
        "llm_provider": llm_provider,
        "ground_truth_atoms": total_gt,
        "results": results,
        "shortcut_flags": dict(shortcut_flags or {}),
        "declared_shortcuts": list(declared_shortcuts or []),
        "policy": {
            "anti_shortcut": _report_to_dict(anti_shortcut),
            "variants": policy_variants,
        },
    }
    if postprocess_summary:
        report["postprocess"] = {
            "enabled": True,
            "dataset": profile_dataset,
            **postprocess_summary,
        }

    with (output_dir / "summary.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    return report
