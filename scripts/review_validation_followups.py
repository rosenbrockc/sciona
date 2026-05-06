#!/usr/bin/env python3
"""Build actionable follow-up queues from validation result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sciona.architect.validation_followup import (  # noqa: E402
    build_validation_followup_report_from_paths,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="Validation result JSON files")
    parser.add_argument("--output", required=True, help="Path for follow-up report JSON")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--similarity-threshold", type=float, default=0.45)
    parser.add_argument("--max-tickets", type=int, default=100)
    parser.add_argument("--max-clusters", type=int, default=50)
    args = parser.parse_args()

    report = build_validation_followup_report_from_paths(
        args.results,
        min_support=args.min_support,
        similarity_threshold=args.similarity_threshold,
        max_tickets=args.max_tickets,
        max_clusters=args.max_clusters,
    )
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")

    print(f"reviewed {report.total_results} validation result(s)")
    print(f"strict assessments: {report.assessment_counts}")
    print(f"rescued_by_expansion: {report.rescued_by_expansion_count}")
    print(f"trick_review_tickets: {report.trick_review_ticket_count}")
    print(f"remaining_divergent: {report.divergent_count}")
    print(
        "divergent_gap_clusters: "
        f"{len(report.divergent_gap_report.clusters)} "
        f"(candidate={report.divergent_gap_report.reusable_candidate_count}, "
        f"existing={report.divergent_gap_report.existing_operation_count}, "
        f"one_off={report.divergent_gap_report.one_off_count})"
    )
    if report.divergent_family_counts:
        print("top_divergent_families:")
        for family, count in list(report.divergent_family_counts.items())[:10]:
            print(f"  {family}: {count}")


if __name__ == "__main__":
    main()
