#!/usr/bin/env python
"""Validate symbolic physics fixtures and PDG CDG publication rows offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sciona.physics_ingest.validation import (  # noqa: E402
    build_physics_ingestion_validation_report,
    discover_changed_pdg_payload_fixture_paths,
    discover_changed_symbolic_fixture_paths,
    discover_pdg_payload_fixture_paths,
    discover_symbolic_fixture_paths,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate physics symbolic fixtures and PDG CDG rows.",
    )
    parser.add_argument(
        "--atoms-repo",
        type=Path,
        default=Path("../sciona-atoms-physics"),
        help="Path to sciona-atoms-physics for fixture discovery and live manifest checks.",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        type=Path,
        default=[],
        help="Specific symbolic publication fixture to validate. May be repeated.",
    )
    parser.add_argument(
        "--pdg-json",
        action="append",
        type=Path,
        default=[],
        help="Specific PDG payload JSON file to validate. May be repeated.",
    )
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help=(
            "Validate only git-changed discovered fixture files. Explicit "
            "--fixture and --pdg-json paths are still validated; source "
            "execution and adapter coverage checks remain enabled unless "
            "their skip flags are passed."
        ),
    )
    parser.add_argument(
        "--skip-atoms",
        action="store_true",
        help="Skip symbolic publication fixture validation.",
    )
    parser.add_argument(
        "--skip-pdg",
        action="store_true",
        help="Skip the built-in PDG validation fixture and any --pdg-json payloads.",
    )
    parser.add_argument(
        "--skip-source-execution",
        action="store_true",
        help="Skip source retrieval execution readiness validation.",
    )
    parser.add_argument(
        "--skip-source-adapter-coverage",
        action="store_true",
        help="Skip source adapter coverage validation.",
    )
    parser.add_argument(
        "--skip-source-adapter-data-artifact-seeds",
        action="store_true",
        help="Skip source adapter data-artifact seed quality validation.",
    )
    parser.add_argument(
        "--source-max-jobs",
        type=int,
        default=None,
        help="Limit source retrieval jobs included in the execution readiness check.",
    )
    parser.add_argument(
        "--source-job-id",
        action="append",
        default=[],
        help="Restrict source execution readiness to a retrieval job id. May be repeated.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when expected local fixture inventories are absent.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit only machine-readable JSON.",
    )
    args = parser.parse_args(argv)

    fixture_paths = tuple(args.fixture)
    pdg_payload_paths = tuple(args.pdg_json)
    atoms_repo = None if args.skip_atoms else args.atoms_repo
    if not args.skip_atoms:
        if args.changed_only:
            fixture_paths = _unique_paths(
                (
                    *fixture_paths,
                    *discover_changed_symbolic_fixture_paths(args.atoms_repo),
                )
            )
        elif not fixture_paths and args.atoms_repo.exists():
            fixture_paths = discover_symbolic_fixture_paths(args.atoms_repo)
    if not args.skip_pdg:
        if args.changed_only:
            pdg_payload_paths = _unique_paths(
                (
                    *pdg_payload_paths,
                    *discover_changed_pdg_payload_fixture_paths(REPO_ROOT),
                )
            )
        elif not pdg_payload_paths:
            pdg_payload_paths = discover_pdg_payload_fixture_paths(REPO_ROOT)

    # Changed-only narrows fixture and payload discovery only. Source execution
    # readiness plus adapter contracts are fast global checks, so they stay on
    # unless the caller uses the existing skip flags.

    report = build_physics_ingestion_validation_report(
        fixture_paths=fixture_paths,
        atoms_repo=atoms_repo,
        pdg_payload_paths=() if args.skip_pdg else pdg_payload_paths,
        include_default_pdg=not args.skip_pdg,
        include_source_execution=not args.skip_source_execution,
        include_source_adapter_coverage=not args.skip_source_adapter_coverage,
        include_source_adapter_data_artifact_seeds=(
            not args.skip_source_adapter_data_artifact_seeds
        ),
        source_max_jobs=args.source_max_jobs,
        source_job_id=tuple(args.source_job_id) or None,
        strict=args.strict,
    )

    if args.json:
        print(json.dumps(report, sort_keys=True), file=sys.stdout)
    else:
        _print_text_report(report)
    return 0 if report["ok"] else 1


def _unique_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)
    return tuple(unique_paths)


def _print_text_report(report: dict[str, object]) -> None:
    summary = report["summary"]
    assert isinstance(summary, dict)
    print("PHYSICS INGESTION VALIDATION REPORT")
    print(f"ok: {report['ok']}")
    print(
        "checks: "
        f"{summary['check_count']} total, "
        f"{summary['failed_check_count']} failed, "
        f"{summary['error_count']} errors"
    )
    for check in report["checks"]:
        assert isinstance(check, dict)
        status = "ok" if check["ok"] else "failed"
        print(f"- {check['check_id']} {status}: {check['subject']}")
        for issue in check["issues"]:
            assert isinstance(issue, dict)
            print(
                "  "
                f"{issue['severity']} "
                f"{issue['reason']} "
                f"{issue['table']} "
                f"{issue['subject']} "
                f"{issue['detail']}".rstrip()
            )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
