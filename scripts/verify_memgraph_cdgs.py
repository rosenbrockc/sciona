#!/usr/bin/env python3
"""Verify all CDGs in Memgraph conform to the schema expected by the
isomorphism search (graph_retrieval layers 1-3) and are trustworthy
for downstream use.

Checks performed:
  1. Schema: every Atom has required properties with correct types
  2. Labels: status labels (:Atomic/:Decomposed) match .status property
  3. Topo-hash: every Decomposed atom with children has a valid topo_hash
  4. Topo-hash correctness: recomputed hash matches stored hash
  5. Port consistency: n_inputs/n_outputs match actual HAS_INPUT/HAS_OUTPUT
  6. PARENT_OF integrity: children reachable, no orphan atoms
  7. DATA_FLOW integrity: edges connect siblings under the same parent
  8. Description quality: non-empty, minimum length, no raw code / jargon
  9. Concept-type validity: values belong to the known enum
  10. Uniqueness: no duplicate fqn values
  11. Per-repo summary with aggregate health score

Usage:
    python scripts/verify_memgraph_cdgs.py [--uri bolt://localhost:7687]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Known valid concept types (from ageom.architect.models.ConceptType)
# ---------------------------------------------------------------------------
VALID_CONCEPT_TYPES = {
    "sorting", "searching", "divide_and_conquer", "greedy",
    "dynamic_programming", "graph_traversal", "graph_optimization",
    "string_matching", "geometry", "arithmetic", "number_theory",
    "combinatorics", "algebra", "analysis", "set_theory",
    "signal_transform", "signal_filter", "graph_signal_processing",
    "neural_network", "sampler", "log_prob", "posterior_update",
    "variational_inference", "prior_init", "prior_distribution",
    "likelihood_evaluation", "probabilistic_oracle", "oracle_gradient",
    "mcmc_kernel", "mcmc_proposal", "vi_elbo", "sequential_filter",
    "smc_reweight", "message_passing", "conjugate_update",
    "custom", "external_tool",
}

# Jargon / suspicious description patterns
_JARGON_PATTERNS = [
    "def ",       # raw code leaked into description
    "import ",    # raw code
    "return ",    # raw code
    "self.",      # raw code
    "TODO",       # placeholder
    "FIXME",      # placeholder
    "NotImplemented",
]

# Required Atom properties and their expected types
_REQUIRED_ATOM_PROPS = {
    "node_id": str,
    "fqn": str,
    "repo": str,
    "name": str,
    "description": str,
    "concept_type": str,
    "status": str,
    "depth": int,
    "n_inputs": int,
    "n_outputs": int,
}


@dataclass
class Issue:
    severity: str  # "ERROR" or "WARN"
    repo: str
    fqn: str
    check: str
    detail: str


@dataclass
class VerificationReport:
    total_atoms: int = 0
    total_decomposed: int = 0
    total_atomic: int = 0
    total_edges_data_flow: int = 0
    total_edges_parent_of: int = 0
    total_input_ports: int = 0
    total_output_ports: int = 0
    repos: set = field(default_factory=set)
    issues: list[Issue] = field(default_factory=list)

    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "WARN"]


def _topo_hash_from_rows(
    children: list[dict], edges: list[dict],
) -> str:
    """Recompute topo_hash from child nodes and their DATA_FLOW edges."""
    child_ids = {c["node_id"] for c in children}
    degree_seq: list[tuple[int, int]] = []
    for cid in sorted(child_ids):
        in_deg = sum(1 for e in edges if e["target_id"] == cid)
        out_deg = sum(1 for e in edges if e["source_id"] == cid)
        degree_seq.append((in_deg, out_deg))
    raw = str(sorted(degree_seq))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


async def run_verification(uri: str) -> VerificationReport:
    from neo4j import AsyncGraphDatabase

    report = VerificationReport()

    driver = AsyncGraphDatabase.driver(uri)
    try:
        async with driver.session() as session:
            # ---------------------------------------------------------------
            # Fetch all atoms
            # ---------------------------------------------------------------
            result = await session.run(
                "MATCH (a:Atom) "
                "RETURN a, labels(a) AS labels "
                "ORDER BY a.repo, a.fqn"
            )
            atoms_raw = []
            async for rec in result:
                node = dict(rec["a"])
                node["_labels"] = rec["labels"]
                atoms_raw.append(node)

            report.total_atoms = len(atoms_raw)
            fqn_index: dict[str, dict] = {}
            repo_atoms: dict[str, list[dict]] = defaultdict(list)

            for atom in atoms_raw:
                fqn = atom.get("fqn", "")
                repo = atom.get("repo", "")
                report.repos.add(repo)
                repo_atoms[repo].append(atom)

                # --- Check 1: required properties ---
                for prop, expected_type in _REQUIRED_ATOM_PROPS.items():
                    val = atom.get(prop)
                    if val is None:
                        report.issues.append(Issue(
                            "ERROR", repo, fqn, "missing_property",
                            f"Missing required property: {prop}",
                        ))
                    elif not isinstance(val, expected_type):
                        report.issues.append(Issue(
                            "ERROR", repo, fqn, "wrong_type",
                            f"{prop}: expected {expected_type.__name__}, "
                            f"got {type(val).__name__}",
                        ))

                # --- Check 10: uniqueness ---
                if fqn in fqn_index:
                    report.issues.append(Issue(
                        "ERROR", repo, fqn, "duplicate_fqn",
                        f"Duplicate fqn (also in {fqn_index[fqn].get('repo')})",
                    ))
                fqn_index[fqn] = atom

                status = atom.get("status", "")
                labels = set(atom.get("_labels", []))

                # --- Check 2: label consistency ---
                if status == "atomic":
                    report.total_atomic += 1
                    if "Atomic" not in labels:
                        report.issues.append(Issue(
                            "ERROR", repo, fqn, "label_mismatch",
                            "status='atomic' but missing :Atomic label",
                        ))
                    if "Decomposed" in labels:
                        report.issues.append(Issue(
                            "ERROR", repo, fqn, "label_mismatch",
                            "status='atomic' but has :Decomposed label",
                        ))
                elif status == "decomposed":
                    report.total_decomposed += 1
                    if "Decomposed" not in labels:
                        report.issues.append(Issue(
                            "ERROR", repo, fqn, "label_mismatch",
                            "status='decomposed' but missing :Decomposed label",
                        ))
                    if "Atomic" in labels:
                        report.issues.append(Issue(
                            "ERROR", repo, fqn, "label_mismatch",
                            "status='decomposed' but has :Atomic label",
                        ))
                else:
                    report.issues.append(Issue(
                        "WARN", repo, fqn, "unknown_status",
                        f"Unknown status: '{status}'",
                    ))

                # --- Check 9: concept_type validity ---
                ct = atom.get("concept_type", "")
                if ct and ct not in VALID_CONCEPT_TYPES:
                    report.issues.append(Issue(
                        "WARN", repo, fqn, "unknown_concept_type",
                        f"concept_type '{ct}' not in known enum",
                    ))

                # --- Check 8: description quality ---
                desc = atom.get("description", "")
                name = atom.get("name", "")
                if not desc:
                    report.issues.append(Issue(
                        "WARN", repo, fqn, "empty_description",
                        "Description is empty",
                    ))
                elif len(desc) < 10:
                    report.issues.append(Issue(
                        "WARN", repo, fqn, "short_description",
                        f"Description very short ({len(desc)} chars): '{desc}'",
                    ))

                for pattern in _JARGON_PATTERNS:
                    if pattern in desc:
                        report.issues.append(Issue(
                            "WARN", repo, fqn, "description_jargon",
                            f"Description contains suspicious pattern '{pattern}': "
                            f"'{desc[:80]}...'",
                        ))
                        break  # one warning per atom

                if not name:
                    report.issues.append(Issue(
                        "ERROR", repo, fqn, "empty_name",
                        "Name is empty",
                    ))

            # ---------------------------------------------------------------
            # Fetch PARENT_OF edges
            # ---------------------------------------------------------------
            result = await session.run(
                "MATCH (p:Atom)-[:PARENT_OF]->(c:Atom) "
                "RETURN p.fqn AS parent_fqn, p.node_id AS parent_nid, "
                "       p.repo AS repo, "
                "       c.fqn AS child_fqn, c.node_id AS child_nid"
            )
            parent_of_edges: list[dict] = []
            children_of: dict[str, list[str]] = defaultdict(list)
            child_to_parent: dict[str, str] = {}
            async for rec in result:
                row = dict(rec)
                parent_of_edges.append(row)
                children_of[row["parent_fqn"]].append(row["child_fqn"])
                child_to_parent[row["child_fqn"]] = row["parent_fqn"]
            report.total_edges_parent_of = len(parent_of_edges)

            # --- Check 6: decomposed atoms must have children ---
            for atom in atoms_raw:
                fqn = atom.get("fqn", "")
                repo = atom.get("repo", "")
                if atom.get("status") == "decomposed":
                    if fqn not in children_of or len(children_of[fqn]) == 0:
                        report.issues.append(Issue(
                            "ERROR", repo, fqn, "decomposed_no_children",
                            "Decomposed atom has no PARENT_OF children",
                        ))

            # --- Check 6b: orphan atoms (not root, no parent) ---
            all_fqns = set(fqn_index.keys())
            atoms_with_parent = set(child_to_parent.keys())
            root_fqns = {
                a.get("fqn", "")
                for a in atoms_raw
                if a.get("depth", 0) == 0
            }
            for fqn in all_fqns:
                if fqn not in atoms_with_parent and fqn not in root_fqns:
                    atom = fqn_index[fqn]
                    # Only flag if depth > 0 (non-root should have a parent)
                    if atom.get("depth", 0) > 0:
                        report.issues.append(Issue(
                            "WARN", atom.get("repo", ""), fqn,
                            "orphan_atom",
                            f"Atom at depth {atom.get('depth')} has no "
                            f"PARENT_OF relationship",
                        ))

            # ---------------------------------------------------------------
            # Fetch DATA_FLOW edges
            # ---------------------------------------------------------------
            result = await session.run(
                "MATCH (s:Atom)-[r:DATA_FLOW]->(t:Atom) "
                "RETURN s.fqn AS source_fqn, s.node_id AS source_nid, "
                "       t.fqn AS target_fqn, t.node_id AS target_nid, "
                "       s.repo AS repo, "
                "       r.output_name AS output_name, "
                "       r.input_name AS input_name, "
                "       r.source_type AS source_type, "
                "       r.target_type AS target_type"
            )
            data_flow_edges: list[dict] = []
            df_edges_by_repo: dict[str, list[dict]] = defaultdict(list)
            async for rec in result:
                row = dict(rec)
                data_flow_edges.append(row)
                df_edges_by_repo[row["repo"]].append(row)
            report.total_edges_data_flow = len(data_flow_edges)

            # --- Check 7: DATA_FLOW edges connect siblings ---
            for edge in data_flow_edges:
                src_parent = child_to_parent.get(edge["source_fqn"])
                tgt_parent = child_to_parent.get(edge["target_fqn"])
                repo = edge["repo"]
                if src_parent and tgt_parent and src_parent != tgt_parent:
                    report.issues.append(Issue(
                        "ERROR", repo,
                        f"{edge['source_fqn']}->{edge['target_fqn']}",
                        "cross_parent_data_flow",
                        f"DATA_FLOW crosses parent boundary: "
                        f"source parent={src_parent}, target parent={tgt_parent}",
                    ))

                # Check edge properties not empty
                if not edge.get("output_name"):
                    report.issues.append(Issue(
                        "WARN", repo,
                        f"{edge['source_fqn']}->{edge['target_fqn']}",
                        "empty_output_name",
                        "DATA_FLOW edge has empty output_name",
                    ))
                if not edge.get("input_name"):
                    report.issues.append(Issue(
                        "WARN", repo,
                        f"{edge['source_fqn']}->{edge['target_fqn']}",
                        "empty_input_name",
                        "DATA_FLOW edge has empty input_name",
                    ))

            # ---------------------------------------------------------------
            # Fetch port counts
            # ---------------------------------------------------------------
            result = await session.run(
                "MATCH (a:Atom)-[:HAS_INPUT]->(p:InputPort) "
                "RETURN a.fqn AS fqn, a.repo AS repo, "
                "       a.n_inputs AS declared, count(p) AS actual"
            )
            async for rec in result:
                report.total_input_ports += rec["actual"]
                if rec["declared"] != rec["actual"]:
                    report.issues.append(Issue(
                        "ERROR", rec["repo"], rec["fqn"],
                        "input_port_mismatch",
                        f"n_inputs={rec['declared']} but "
                        f"{rec['actual']} HAS_INPUT edges",
                    ))

            result = await session.run(
                "MATCH (a:Atom)-[:HAS_OUTPUT]->(p:OutputPort) "
                "RETURN a.fqn AS fqn, a.repo AS repo, "
                "       a.n_outputs AS declared, count(p) AS actual"
            )
            async for rec in result:
                report.total_output_ports += rec["actual"]
                if rec["declared"] != rec["actual"]:
                    report.issues.append(Issue(
                        "ERROR", rec["repo"], rec["fqn"],
                        "output_port_mismatch",
                        f"n_outputs={rec['declared']} but "
                        f"{rec['actual']} HAS_OUTPUT edges",
                    ))

            # --- Check 5b: atoms with n_inputs>0 but no ports ---
            atoms_with_input_ports = set()
            result2 = await session.run(
                "MATCH (a:Atom)-[:HAS_INPUT]->() "
                "RETURN DISTINCT a.fqn AS fqn"
            )
            async for rec in result2:
                atoms_with_input_ports.add(rec["fqn"])

            atoms_with_output_ports = set()
            result3 = await session.run(
                "MATCH (a:Atom)-[:HAS_OUTPUT]->() "
                "RETURN DISTINCT a.fqn AS fqn"
            )
            async for rec in result3:
                atoms_with_output_ports.add(rec["fqn"])

            for atom in atoms_raw:
                fqn = atom.get("fqn", "")
                repo = atom.get("repo", "")
                ni = atom.get("n_inputs", 0)
                no = atom.get("n_outputs", 0)
                if ni > 0 and fqn not in atoms_with_input_ports:
                    report.issues.append(Issue(
                        "ERROR", repo, fqn, "missing_input_ports",
                        f"n_inputs={ni} but no HAS_INPUT edges exist",
                    ))
                if no > 0 and fqn not in atoms_with_output_ports:
                    report.issues.append(Issue(
                        "ERROR", repo, fqn, "missing_output_ports",
                        f"n_outputs={no} but no HAS_OUTPUT edges exist",
                    ))

            # ---------------------------------------------------------------
            # Check 3 & 4: topo_hash for Decomposed atoms
            # ---------------------------------------------------------------
            for atom in atoms_raw:
                fqn = atom.get("fqn", "")
                repo = atom.get("repo", "")
                if atom.get("status") != "decomposed":
                    continue

                stored_hash = atom.get("topo_hash", "")
                child_fqns = children_of.get(fqn, [])

                if not child_fqns:
                    continue  # already flagged as decomposed_no_children

                # Recompute topo_hash
                children_data = []
                for cf in child_fqns:
                    ca = fqn_index.get(cf)
                    if ca:
                        children_data.append({"node_id": ca["node_id"]})

                # Collect DATA_FLOW edges among these children
                child_fqn_set = set(child_fqns)
                sibling_edges = [
                    {"source_id": e["source_nid"], "target_id": e["target_nid"]}
                    for e in df_edges_by_repo.get(repo, [])
                    if e["source_fqn"] in child_fqn_set
                    and e["target_fqn"] in child_fqn_set
                ]

                recomputed = _topo_hash_from_rows(children_data, sibling_edges)

                if not stored_hash:
                    report.issues.append(Issue(
                        "ERROR", repo, fqn, "missing_topo_hash",
                        f"Decomposed atom with {len(child_fqns)} children "
                        f"has no topo_hash (should be {recomputed})",
                    ))
                elif len(stored_hash) != 16:
                    report.issues.append(Issue(
                        "ERROR", repo, fqn, "invalid_topo_hash",
                        f"topo_hash '{stored_hash}' is not 16 chars",
                    ))
                elif stored_hash != recomputed:
                    report.issues.append(Issue(
                        "ERROR", repo, fqn, "topo_hash_mismatch",
                        f"Stored topo_hash={stored_hash} != "
                        f"recomputed={recomputed}",
                    ))

    finally:
        await driver.close()

    return report


def print_report(report: VerificationReport) -> None:
    errors = report.errors()
    warnings = report.warnings()

    print("=" * 72)
    print("MEMGRAPH CDG VERIFICATION REPORT")
    print("=" * 72)
    print()
    print(f"  Repos:           {len(report.repos)}")
    print(f"  Total atoms:     {report.total_atoms}")
    print(f"    Decomposed:    {report.total_decomposed}")
    print(f"    Atomic:        {report.total_atomic}")
    print(f"  PARENT_OF edges: {report.total_edges_parent_of}")
    print(f"  DATA_FLOW edges: {report.total_edges_data_flow}")
    print(f"  Input ports:     {report.total_input_ports}")
    print(f"  Output ports:    {report.total_output_ports}")
    print()

    # Per-repo summary
    repo_issues: dict[str, list[Issue]] = defaultdict(list)
    for issue in report.issues:
        repo_issues[issue.repo].append(issue)

    print("-" * 72)
    print(f"{'Repo':<45} {'Errors':>7} {'Warns':>7}")
    print("-" * 72)
    for repo in sorted(report.repos):
        issues = repo_issues.get(repo, [])
        nerr = sum(1 for i in issues if i.severity == "ERROR")
        nwarn = sum(1 for i in issues if i.severity == "WARN")
        marker = "  FAIL" if nerr > 0 else "  OK"
        print(f"  {repo:<43} {nerr:>7} {nwarn:>7}{marker}")
    print("-" * 72)
    print()

    # Issue breakdown by check
    check_counts: Counter = Counter()
    for issue in report.issues:
        check_counts[f"{issue.severity}:{issue.check}"] += 1

    if check_counts:
        print("Issue breakdown:")
        for key, count in check_counts.most_common():
            print(f"  {key:<45} {count:>5}")
        print()

    # Print all errors (always)
    if errors:
        print(f"ERRORS ({len(errors)}):")
        print("-" * 72)
        for i in errors[:100]:  # cap output
            print(f"  [{i.repo}] {i.fqn}")
            print(f"    {i.check}: {i.detail}")
        if len(errors) > 100:
            print(f"  ... and {len(errors) - 100} more errors")
        print()

    # Print warnings (capped)
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        print("-" * 72)
        for i in warnings[:50]:
            print(f"  [{i.repo}] {i.fqn}")
            print(f"    {i.check}: {i.detail}")
        if len(warnings) > 50:
            print(f"  ... and {len(warnings) - 50} more warnings")
        print()

    # Final verdict
    print("=" * 72)
    if not errors and not warnings:
        print("RESULT: ALL CHECKS PASSED")
    elif not errors:
        print(f"RESULT: PASSED with {len(warnings)} warning(s)")
    else:
        print(f"RESULT: FAILED — {len(errors)} error(s), {len(warnings)} warning(s)")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Verify Memgraph CDGs for isomorphism search readiness"
    )
    parser.add_argument(
        "--uri", default="bolt://localhost:7687",
        help="Memgraph bolt URI (default: bolt://localhost:7687)",
    )
    args = parser.parse_args()

    report = asyncio.run(run_verification(args.uri))
    print_report(report)

    # Exit code: 1 if errors, 0 otherwise
    sys.exit(1 if report.errors() else 0)


if __name__ == "__main__":
    main()
