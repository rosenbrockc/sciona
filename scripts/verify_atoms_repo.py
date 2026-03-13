#!/usr/bin/env python3
"""Verify completeness of an atoms repository such as ageoa."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ageom.atoms_repo_verifier import verify_atoms_repo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify generated atom modules are complete and self-contained."
    )
    parser.add_argument(
        "repo_root",
        help="Path to the atoms repository root (for example ../ageo-atoms).",
    )
    parser.add_argument(
        "--package",
        required=True,
        help="Top-level Python package name inside the repo (for example ageoa).",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--import-smoke",
        action="store_true",
        help="Also attempt module imports with the selected Python executable.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for optional import smoke checks.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = verify_atoms_repo(
        Path(args.repo_root),
        args.package,
        import_smoke=args.import_smoke,
        python_executable=args.python,
    )
    if args.json_output:
        print(report.to_json())
    else:
        print(report.to_text())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
