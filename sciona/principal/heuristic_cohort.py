"""Cross-family cohort building for heuristic-guided proposal selection."""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sciona.principal.datasets import read_adapter
from sciona.principal.models import BenchmarkResult, OptimizationMetric
from sciona.principal.runtime_heuristics import RuntimeHeuristicObservation
from sciona.synthesizer.models import ExportBundle


@dataclass(frozen=True)
class HeuristicCohortMember:
    """One deterministic dataset member used for cohort guidance."""

    tracker_value: str
    member_label: str
    folder_name: str
    tracker_csv: Path


@dataclass(frozen=True)
class MaterializedHeuristicCohort:
    """Temporary dataset root that exposes a deterministic cohort tracker."""

    dataset_root: Path
    adapter_path: Path
    source_tracker_path: Path
    combined_tracker_csv: Path
    members: tuple[HeuristicCohortMember, ...]


def _cohort_tracker_value(size: int, suffix: str = "") -> str:
    base = f"heuristic_cohort_{size}"
    return f"{base}_{suffix}" if suffix else base


def _resolve_meta_source(
    adapter_path: Path,
    varset: dict[str, str] | None,
) -> tuple[str, str]:
    spec = read_adapter(str(adapter_path.parent), adapter_path.stem, varset=varset)
    meta = spec.get("meta", {}) if isinstance(spec, dict) else {}
    source = str(meta.get("source", "") or "").strip()
    folder = meta.get("folder", {})
    folder_key = (
        str(folder.get("source", "") or "").strip()
        if isinstance(folder, dict)
        else ""
    )
    if not source or not folder_key:
        raise ValueError("adapter template must define meta.source and meta.folder.source")
    return source, folder_key


def _load_tracker_rows(
    adapter_root: Path,
    *,
    tracker_source: str,
    cohort_size: int,
) -> tuple[Path, list[dict[str, Any]]]:
    tracker_path = (adapter_root / tracker_source).resolve()
    if not tracker_path.exists():
        raise FileNotFoundError(f"tracker source not found: {tracker_path}")

    frame = pd.read_csv(tracker_path)
    if len(frame.index) < cohort_size:
        for candidate in (adapter_root / "tracker_full.csv", adapter_root / "tracker.csv"):
            if candidate.exists():
                fallback = pd.read_csv(candidate)
                if len(fallback.index) >= cohort_size:
                    tracker_path = candidate.resolve()
                    frame = fallback
                    break
    rows = frame.to_dict(orient="records")
    if not rows:
        raise ValueError(f"tracker source has no rows: {tracker_path}")
    return tracker_path, rows


def materialize_heuristic_tracker_cohort(
    *,
    adapter_path: str,
    varset: dict[str, str] | None,
    cohort_size: int,
    output_root: Path,
) -> MaterializedHeuristicCohort | None:
    """Create a temporary dataset root exposing deterministic cohort trackers."""
    if cohort_size <= 1:
        return None

    adapter = Path(adapter_path).expanduser().resolve()
    output_root = output_root.resolve()
    tracker_source, folder_key = _resolve_meta_source(adapter, varset)
    source_tracker_path, rows = _load_tracker_rows(
        adapter.parent,
        tracker_source=tracker_source,
        cohort_size=cohort_size,
    )

    cohort_root = (output_root / "heuristic_cohort_dataset").resolve()
    if cohort_root.exists():
        shutil.rmtree(cohort_root)
    cohort_root.mkdir(parents=True, exist_ok=True)

    for template_path in adapter.parent.glob("*.yml"):
        target = cohort_root / template_path.name
        if not target.exists():
            target.symlink_to(template_path.resolve())

    selected_rows: list[dict[str, Any]] = []
    members: list[HeuristicCohortMember] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        folder_name = str(row.get(folder_key, "") or "").strip()
        if not folder_name:
            continue
        source_dir = adapter.parent / folder_name
        if not source_dir.exists():
            continue
        target_dir = cohort_root / folder_name
        if not target_dir.exists():
            target_dir.symlink_to(source_dir.resolve(), target_is_directory=True)
        selected_rows.append(dict(row))
        tracker_value = _cohort_tracker_value(cohort_size, f"{index:03d}")
        tracker_csv = cohort_root / f"tracker_{tracker_value}.csv"
        pd.DataFrame([row]).to_csv(tracker_csv, index=False)
        member_label = str(
            row.get("night_id")
            or row.get("trial_name")
            or row.get("session_date")
            or folder_name
        )
        members.append(
            HeuristicCohortMember(
                tracker_value=tracker_value,
                member_label=member_label,
                folder_name=folder_name,
                tracker_csv=tracker_csv,
            )
        )

    if not members:
        return None

    combined_tracker_csv = cohort_root / f"tracker_{_cohort_tracker_value(cohort_size)}.csv"
    pd.DataFrame(selected_rows).to_csv(combined_tracker_csv, index=False)
    return MaterializedHeuristicCohort(
        dataset_root=cohort_root,
        adapter_path=cohort_root / adapter.name,
        source_tracker_path=source_tracker_path,
        combined_tracker_csv=combined_tracker_csv,
        members=tuple(members),
    )


def _shadow_bundle(bundle: ExportBundle, *, output_dir: Path) -> ExportBundle:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return bundle.model_copy(
        update={
            "output_dir": output_dir,
            "source_path": bundle.source_path.resolve(),
            "compiled_artifact": (
                bundle.compiled_artifact.resolve()
                if bundle.compiled_artifact is not None
                else None
            ),
            "executable_artifact": (
                bundle.executable_artifact.resolve()
                if bundle.executable_artifact is not None
                else None
            ),
        }
    )


def summarize_heuristic_cohort(
    entries: list[dict[str, Any]],
    *,
    cohort_size: int,
    source_tracker_path: str,
    combined_tracker_csv: str,
) -> dict[str, Any]:
    """Aggregate heuristic recurrence across cohort members."""
    heuristic_stats: dict[str, dict[str, Any]] = {}
    member_count = len(entries)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        seen_in_member: set[str] = set()
        for item in entry.get("heuristics", []) or []:
            try:
                observation = RuntimeHeuristicObservation.model_validate(item)
                heuristic_id = observation.heuristic.heuristic_id
                confidence = float(observation.confidence)
                source_section = observation.source_section
            except Exception:
                if not isinstance(item, dict):
                    continue
                heuristic = item.get("heuristic", {})
                if not isinstance(heuristic, dict):
                    continue
                heuristic_id = str(heuristic.get("heuristic_id", "") or "").strip()
                if not heuristic_id:
                    continue
                try:
                    confidence = float(
                        item.get("confidence", heuristic.get("confidence", 0.0)) or 0.0
                    )
                except (TypeError, ValueError):
                    confidence = 0.0
                source_section = str(item.get("source_section", "") or "")
            stats = heuristic_stats.setdefault(
                heuristic_id,
                {
                    "occurrence_count": 0,
                    "member_count": 0,
                    "mean_confidence": 0.0,
                    "max_confidence": 0.0,
                    "source_sections": set(),
                    "member_labels": [],
                },
            )
            stats["occurrence_count"] += 1
            stats["mean_confidence"] += confidence
            stats["max_confidence"] = max(
                float(stats["max_confidence"]),
                confidence,
            )
            if source_section:
                stats["source_sections"].add(source_section)
            if heuristic_id not in seen_in_member:
                stats["member_count"] += 1
                stats["member_labels"].append(str(entry.get("member_label", "")))
                seen_in_member.add(heuristic_id)

    serialized: dict[str, Any] = {}
    for heuristic_id, stats in heuristic_stats.items():
        occurrence_count = int(stats["occurrence_count"])
        member_presence = int(stats["member_count"])
        serialized[heuristic_id] = {
            "occurrence_count": occurrence_count,
            "member_count": member_presence,
            "coverage_fraction": (
                float(member_presence / member_count) if member_count > 0 else 0.0
            ),
            "mean_confidence": (
                float(stats["mean_confidence"] / occurrence_count)
                if occurrence_count > 0
                else 0.0
            ),
            "max_confidence": float(stats["max_confidence"]),
            "source_sections": sorted(
                section for section in stats["source_sections"] if section
            ),
            "member_labels": [
                label
                for label in stats["member_labels"]
                if isinstance(label, str) and label
            ][:member_count],
        }

    return {
        "cohort_size": cohort_size,
        "evaluated_member_count": member_count,
        "source_tracker_path": source_tracker_path,
        "combined_tracker_csv": combined_tracker_csv,
        "heuristics": serialized,
        "members": entries,
    }


async def build_adapter_heuristic_cohort(
    *,
    bundle: ExportBundle,
    sandbox: Any,
    adapter_path: str,
    metric: OptimizationMetric,
    dataset_varset: dict[str, str] | None,
    evaluation_spec: dict[str, Any] | str | None,
    cohort_size: int,
    max_concurrency: int = 1,
) -> dict[str, Any] | None:
    """Evaluate one bundle across a deterministic cohort and summarize heuristics."""
    materialized = materialize_heuristic_tracker_cohort(
        adapter_path=adapter_path,
        varset=dataset_varset,
        cohort_size=cohort_size,
        output_root=bundle.output_dir,
    )
    if materialized is None:
        return None

    member_output_root = bundle.output_dir / "heuristic_cohort_runs"
    concurrency = max(1, int(max_concurrency or 1))

    async def _evaluate_member(
        index: int,
        member: HeuristicCohortMember,
    ) -> dict[str, Any]:
        member_bundle = _shadow_bundle(
            bundle,
            output_dir=member_output_root / f"member_{index:03d}",
        )
        member_varset = dict(dataset_varset or {})
        member_varset["tracker"] = member.tracker_value
        result: BenchmarkResult = await sandbox.evaluate_adapter(
            member_bundle,
            str(materialized.adapter_path),
            metric,
            varset=member_varset,
            evaluation_spec=evaluation_spec,
        )
        artifacts = dict(result.runtime_artifacts)
        return {
            "member_index": index,
            "member_label": member.member_label,
            "folder_name": member.folder_name,
            "tracker_value": member.tracker_value,
            "loss": float(result.global_loss),
            "heuristics": list(artifacts.get("heuristics", []) or []),
        }

    entries: list[dict[str, Any]] = []
    skipped_members: list[dict[str, Any]] = []
    members = list(materialized.members)
    next_start = 0
    while next_start < len(members) and len(entries) < cohort_size:
        batch = members[next_start : next_start + concurrency]
        batch_results = await asyncio.gather(
            *[
                _evaluate_member(index, member)
                for index, member in enumerate(
                    batch,
                    start=next_start + 1,
                )
            ]
        )
        batch_results.sort(key=lambda item: int(item.get("member_index", 0)))
        for entry in batch_results:
            if entry["loss"] >= 1e11 or not entry["heuristics"]:
                skipped_members.append(entry)
                continue
            entries.append(entry)
            if len(entries) >= cohort_size:
                break
        next_start += len(batch)

    summary = summarize_heuristic_cohort(
        entries,
        cohort_size=cohort_size,
        source_tracker_path=str(materialized.source_tracker_path),
        combined_tracker_csv=str(materialized.combined_tracker_csv),
    )
    summary["attempted_member_count"] = len(entries) + len(skipped_members)
    summary["skipped_members"] = skipped_members[:10]
    return summary
