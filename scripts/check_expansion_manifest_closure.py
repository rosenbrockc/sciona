#!/usr/bin/env python3
"""Release gate for expansion/refinement inventory closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sciona.principal.expansion_manifest import (  # noqa: E402
    build_expansion_inventory_manifest,
    check_expansion_manifest_closure,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Optional path for the closure report JSON")
    parser.add_argument("--manifest-output", help="Optional path for the common inventory manifest JSON")
    args = parser.parse_args()

    report = check_expansion_manifest_closure()
    payload = report.to_dict()
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    if args.manifest_output:
        manifest = build_expansion_inventory_manifest()
        Path(args.manifest_output).write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"expansion_manifest_closure ok={report.ok} "
        f"assets={report.asset_count} operations={report.operation_count}"
    )
    if report.missing_provider_inventory_roots:
        print("missing provider inventories:")
        for root in report.missing_provider_inventory_roots:
            print(f"  {root}")
    if report.missing_asset_backed_rule_sets:
        print("missing asset-backed rule sets:")
        for family in report.missing_asset_backed_rule_sets:
            print(f"  {family}")
    if report.missing_runtime_rules:
        print("missing runtime rules:")
        for rule in report.missing_runtime_rules:
            print(f"  {rule}")
    if report.manifest_sink_mismatches:
        print("manifest sink mismatches:")
        for sink in report.manifest_sink_mismatches:
            print(f"  {sink}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

