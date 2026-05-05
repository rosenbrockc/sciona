#!/usr/bin/env python3
"""Mine reusable expansion/refinement gaps from validation result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sciona.architect.expansion_gap_mining import (  # noqa: E402
    load_validation_results,
    mine_expansion_gaps,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="Validation result JSON files to mine")
    parser.add_argument("--output", required=True, help="Path for the mining report JSON")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--similarity-threshold", type=float, default=0.45)
    parser.add_argument("--max-clusters", type=int, default=50)
    parser.add_argument(
        "--include-assessment",
        action="append",
        default=None,
        help=(
            "Assessment to include. May be repeated. Defaults to partial, "
            "divergent, inadequate, no_template, and no_evaluation."
        ),
    )
    args = parser.parse_args()

    results = load_validation_results(args.results)
    report = mine_expansion_gaps(
        results,
        min_support=args.min_support,
        similarity_threshold=args.similarity_threshold,
        include_assessments=args.include_assessment
        or ("partial", "divergent", "inadequate", "no_template", "no_evaluation"),
        max_clusters=args.max_clusters,
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    print(
        "mined "
        f"{report.occurrence_count} missing technique occurrence(s) into "
        f"{len(report.clusters)} cluster(s)"
    )
    print(f"candidate_reusable_operation: {report.reusable_candidate_count}")
    print(f"covered_by_existing_operation: {report.existing_operation_count}")
    print(f"defer_one_off: {report.one_off_count}")


if __name__ == "__main__":
    main()

