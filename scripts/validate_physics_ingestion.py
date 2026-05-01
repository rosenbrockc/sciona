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
    atoms_repo = None if args.skip_atoms else args.atoms_repo
    if not args.skip_atoms and not fixture_paths and args.atoms_repo.exists():
        fixture_paths = discover_symbolic_fixture_paths(args.atoms_repo)

    report = build_physics_ingestion_validation_report(
        fixture_paths=fixture_paths,
        atoms_repo=atoms_repo,
        pdg_payload_paths=() if args.skip_pdg else tuple(args.pdg_json),
        include_default_pdg=not args.skip_pdg,
        strict=args.strict,
    )

    if args.json:
        print(json.dumps(report, sort_keys=True), file=sys.stdout)
    else:
        _print_text_report(report)
    return 0 if report["ok"] else 1


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
