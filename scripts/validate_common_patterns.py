#!/usr/bin/env python3
"""Validate common_patterns consistency across all atom repos.

Checks:
1. Self-inclusion: declaring atom is in its own pattern's atoms list
2. Symmetry: if A references B in pattern P, B must also declare P referencing A
3. Existence: every atom in every pattern exists in some cdg.json
4. Consistency: all atoms in pattern P declare the same atoms list
5. No orphans: patterns with < 2 atoms are invalid
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    repos = [
        Path.home() / "personal" / name
        for name in [
            "sciona-atoms",
            "sciona-atoms-ml",
            "sciona-atoms-dl",
            "sciona-atoms-bio",
            "sciona-atoms-physics",
            "sciona-atoms-signal",
            "sciona-atoms-cs",
            "sciona-atoms-geo",
            "sciona-atoms-fintech",
            "sciona-atoms-robotics",
        ]
    ]

    # Phase 1: collect all atoms and their patterns
    all_atoms: set[str] = set()
    # pattern_id -> {declaring_atom -> sorted atoms list}
    pattern_declarations: dict[str, dict[str, list[str]]] = defaultdict(dict)
    errors: list[str] = []

    for repo in repos:
        if not repo.exists():
            continue
        for cdg_path in repo.rglob("cdg.json"):
            if "solution_cdgs" in str(cdg_path):
                continue
            try:
                data = json.loads(cdg_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            for node in data.get("nodes", []):
                if node.get("status") != "atomic":
                    continue
                atom_name = node["node_id"]
                all_atoms.add(atom_name)

                for pattern in node.get("common_patterns", []):
                    pid = pattern.get("pattern_id", "")
                    atoms_list = pattern.get("atoms", [])

                    if not pid:
                        errors.append(
                            f"MISSING_ID: {atom_name} in {cdg_path} "
                            f"has a pattern with no pattern_id"
                        )
                        continue

                    # Check 1: self-inclusion
                    if atom_name not in atoms_list:
                        errors.append(
                            f"SELF_INCLUSION: {atom_name} declares pattern "
                            f"'{pid}' but is not in its atoms list: {atoms_list}"
                        )

                    # Check 5: no orphans
                    if len(atoms_list) < 2:
                        errors.append(
                            f"ORPHAN: {atom_name} declares pattern '{pid}' "
                            f"with only {len(atoms_list)} atom(s) — "
                            f"use aliases instead"
                        )

                    pattern_declarations[pid][atom_name] = sorted(atoms_list)

    # Phase 2: cross-atom checks
    for pid, declarations in pattern_declarations.items():
        all_pattern_atoms: set[str] = set()
        for atoms_list in declarations.values():
            all_pattern_atoms.update(atoms_list)

        # Check 3: existence
        for atom in all_pattern_atoms:
            if atom not in all_atoms:
                errors.append(
                    f"EXISTENCE: pattern '{pid}' references atom '{atom}' "
                    f"which does not exist in any cdg.json"
                )

        # Check 2: symmetry
        for atom in all_pattern_atoms:
            if atom not in declarations:
                declared_by = sorted(declarations.keys())
                errors.append(
                    f"SYMMETRY: atom '{atom}' is referenced in pattern "
                    f"'{pid}' (declared by {declared_by}) but does not "
                    f"declare the pattern itself"
                )

        # Check 4: consistency
        canonical: list[str] | None = None
        for declaring_atom, atoms_list in declarations.items():
            if canonical is None:
                canonical = atoms_list
            elif atoms_list != canonical:
                errors.append(
                    f"CONSISTENCY: pattern '{pid}' has inconsistent atoms "
                    f"lists: {declaring_atom} declares {atoms_list} but "
                    f"expected {canonical}"
                )

    # Report
    total_patterns = len(pattern_declarations)
    total_atoms_with_patterns = len(
        {atom for decls in pattern_declarations.values() for atom in decls}
    )

    print(f"Scanned {len(all_atoms)} atoms across {sum(1 for r in repos if r.exists())} repos")
    print(f"Found {total_patterns} patterns declared by {total_atoms_with_patterns} atoms")
    print()

    if errors:
        print(f"ERRORS ({len(errors)}):")
        for err in sorted(errors):
            print(f"  {err}")
        return 1

    print("All pattern validations passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
