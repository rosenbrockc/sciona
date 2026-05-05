#!/usr/bin/env python
"""Promote selected remote PDG derivations through symbolic and CDG rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping
from uuid import NAMESPACE_URL, uuid5

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.physics_inventory_pdg_remote import (  # noqa: E402
    CORE_CYPHER_PATHS,
    DEFAULT_REF,
    DEFAULT_REPO,
    RateLimitedGitHubClient,
    _decode_github_content,
    _github_api_url,
    _github_contents_url,
    _operation_kind,
)


DEFAULT_DERIVATION_IDS = ("000008", "000009", "000018", "129143", "000004")
DEFAULT_USER_AGENT = "sciona-physics-ingest-pdg-remote-wave/0.1"
DEFAULT_TABLE_MODES = {
    "physics_ingest_snapshots": "upsert",
    "physics_equation_candidates": "upsert",
    "artifacts": "upsert",
    "artifact_versions": "upsert",
    "artifact_symbolic_expressions": "upsert",
    "artifact_relationships": "upsert",
    "artifact_cdg_nodes": "upsert",
    "artifact_cdg_edges": "upsert",
    "artifact_cdg_bindings": "upsert",
}

_NODE_RE = re.compile(
    r'UNWIND \[\{id:"(?P<id>[^"]+)",\s+properties:\{(?P<properties>.*?)\}\}\] AS row\s+'
    r"CREATE \(n:(?P<label>\w+)\{id: row\.id\}\)",
    re.DOTALL,
)
_RELATION_RE = re.compile(
    r'UNWIND \[\{start: \{id:"(?P<start>[^"]+)"\}, '
    r'end: \{id:"(?P<end>[^"]+)"\}, properties:\{(?P<props>.*?)\}\}\] AS row\s+'
    r"MATCH \(start:(?P<start_label>\w+).*?\s+"
    r"MATCH \(end:(?P<end_label>\w+).*?\s+"
    r"CREATE \(start\)-\[r:(?P<rel>\w+)\]->\(end\)",
    re.DOTALL,
)
_STRING_FIELD_RE = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*):"(?P<value>(?:[^"\\]|\\.)*)"')
_NUMERIC_FIELD_RE = re.compile(r"(?P<key>[a-zA-Z_][a-zA-Z0-9_]*):(?P<value>-?\d+(?:\.\d+)?)")
_ARTIFACT_NAMESPACE = uuid5(NAMESPACE_URL, "sciona.physics_ingest.pdg_remote_wave.artifact")
_VERSION_NAMESPACE = uuid5(NAMESPACE_URL, "sciona.physics_ingest.pdg_remote_wave.version")
_EXPRESSION_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "sciona.physics_ingest.pdg_remote_wave.expression",
)
_RELATIONSHIP_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "sciona.physics_ingest.pdg_remote_wave.relationship",
)


@dataclass(frozen=True)
class CypherNode:
    node_id: str
    label: str
    properties: Mapping[str, Any]


@dataclass(frozen=True)
class CypherRelation:
    source_id: str
    target_id: str
    source_label: str
    target_label: str
    relation: str
    properties: Mapping[str, Any]


@dataclass(frozen=True)
class PDGRemoteGraph:
    nodes_by_label: Mapping[str, Mapping[str, CypherNode]]
    relations: tuple[CypherRelation, ...]
    core_file_sha256: Mapping[str, str]


@dataclass(frozen=True)
class StepRecord:
    step_id: str
    derivation_id: str
    sequence_index: int
    rule_id: str = ""
    input_expression_ids: tuple[str, ...] = ()
    output_expression_ids: tuple[str, ...] = ()
    feed_ids: tuple[str, ...] = ()


def main() -> int:
    from sciona.physics_ingest.ids import plan_source_bundle_ids
    from sciona.physics_ingest.pdg_cdg import (
        PDGCDGArtifactEnvelope,
        build_pdg_publication_write_rows,
        build_pdg_relationship_ingest,
        validate_pdg_cdg_publication_graph,
    )
    from sciona.physics_ingest.pipeline import run_physics_publication_pipeline
    from sciona.physics_ingest.sources.pdg import parse_pdg_document
    from sciona.physics_ingest.write_plan import (
        build_publication_write_plan,
        merge_publication_insert_rows,
    )

    parser = argparse.ArgumentParser(
        description="Ingest selected remote PDG derivations through CDG rows."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument(
        "--derivation-id",
        action="append",
        default=None,
        help="PDG derivation id to promote. May be supplied multiple times.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for generated write artifacts.",
    )
    parser.add_argument(
        "--source-version",
        default="pdg-remote-wave1-2026-05-05",
        help="Source version string for the remote PDG snapshot.",
    )
    parser.add_argument(
        "--license-expression",
        default="CC-BY-4.0",
        help="License expression recorded in the source snapshot.",
    )
    parser.add_argument(
        "--license-evidence-url",
        default="https://derivationmap.net/developer_documentation",
        help="License/provenance URL recorded in source payloads.",
    )
    parser.add_argument(
        "--min-delay-seconds",
        type=float,
        default=2.0,
        help="Minimum delay between GitHub API requests.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=7,
        help="Hard GitHub API request budget.",
    )
    parser.add_argument(
        "--rate-limit-reserve",
        type=int,
        default=10,
        help="Abort optional fetches when remaining quota reaches this value.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent sent to GitHub.",
    )
    parser.add_argument(
        "--github-token-env",
        default="GITHUB_TOKEN",
        help="Optional env var containing a GitHub token.",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    selected_ids = tuple(args.derivation_id or DEFAULT_DERIVATION_IDS)
    client = RateLimitedGitHubClient(
        user_agent=args.user_agent,
        min_delay_seconds=max(args.min_delay_seconds, 0.0),
        max_requests=max(args.max_requests, 1),
        rate_limit_reserve=max(args.rate_limit_reserve, 0),
        token=str(os.environ.get(args.github_token_env) or ""),
    )
    repo_metadata = client.get_json(_github_api_url(args.repo))
    core_texts: dict[str, str] = {}
    for path in CORE_CYPHER_PATHS:
        if client.should_stop_for_rate_reserve():
            raise RuntimeError("GitHub rate-limit reserve reached before core files loaded")
        content = client.get_json(_github_contents_url(args.repo, path, args.ref))
        core_texts[path] = _decode_github_content(content)

    graph = parse_remote_pdg_cypher(core_texts)
    payloads = build_selected_derivation_payloads(
        graph,
        selected_ids,
        repo=args.repo,
        ref=args.ref,
        source_version=args.source_version,
        license_expression=args.license_expression,
        license_evidence_url=args.license_evidence_url,
    )
    if not payloads:
        raise ValueError(f"no selected derivations were found: {selected_ids}")

    source_equations = _unique_rows_by_id(
        row
        for payload in payloads
        for row in list(payload.get("equations") or [])
        if isinstance(row, Mapping)
    )
    source_edges = [
        dict(row)
        for payload in payloads
        for row in list(payload.get("inference_edges") or [])
        if isinstance(row, Mapping)
    ]
    source_payload = {
        "source_kind": "pdg_remote_cypher_wave",
        "repo": args.repo,
        "ref": args.ref,
        "repo_html_url": str(repo_metadata.get("html_url") or ""),
        "source_version": args.source_version,
        "license_expression": args.license_expression,
        "license_evidence_url": args.license_evidence_url,
        "selected_derivation_ids": list(selected_ids),
        "core_file_sha256": dict(graph.core_file_sha256),
        "equations": source_equations,
        "inference_edges": source_edges,
        "derivations": payloads,
    }
    source_bundle_for_snapshot = parse_pdg_document(
        source_payload,
        source_uri=f"https://github.com/{args.repo}/tree/{args.ref}",
        source_version=args.source_version,
        license_expression=args.license_expression,
        provenance_summary=(
            "Remote Physics Derivation Graph Cypher-derived promotion wave."
        ),
    )
    source_bundle = {
        "bundle_key": f"pdg-remote-wave:{args.repo}:{args.ref}:{args.source_version}",
        "snapshot_row": source_bundle_for_snapshot.snapshot_row,
        "candidate_rows": source_bundle_for_snapshot.candidate_rows(),
    }
    _, planned_source_bundles = plan_source_bundle_ids([source_bundle])
    planned_source_bundle = planned_source_bundles[0]
    candidate_id_by_node_id = {
        str(row["source_candidate_id"]): str(row["candidate_id"])
        for row in planned_source_bundle["candidate_rows"]
    }
    symbolic_rows = _equation_symbolic_rows(
        source_bundle_for_snapshot.equations,
        candidate_id_by_node_id=candidate_id_by_node_id,
        source_version=args.source_version,
        repo=args.repo,
        ref=args.ref,
    )
    expression_bindings = {
        node_id: {
            "expression_id": rows["expression"]["expression_id"],
            "artifact_id": rows["artifact"]["artifact_id"],
            "version_id": rows["version"]["version_id"],
            "label": rows["label"],
            "metadata": {
                "bound_artifact_fqdn": rows["artifact"]["fqdn"],
                "bound_version_content_hash": rows["version"]["content_hash"],
                "binding_confidence": 0.7,
                "binding_source": "pdg_remote_wave",
            },
        }
        for node_id, rows in symbolic_rows.items()
    }

    additional_rows: list[Mapping[str, Iterable[Mapping[str, Any]]]] = []
    relationship_summaries: list[dict[str, Any]] = []
    publication_summaries: list[dict[str, Any]] = []
    graph_diagnostics: list[dict[str, Any]] = []
    for payload in payloads:
        bundle = parse_pdg_document(
            payload,
            source_uri=f"https://github.com/{args.repo}/tree/{args.ref}",
            source_version=args.source_version,
            license_expression=args.license_expression,
            provenance_summary=(
                "Remote Physics Derivation Graph Cypher-derived derivation payload."
            ),
        )
        result = build_pdg_relationship_ingest(
            bundle,
            expression_bindings_by_pdg_node_id=expression_bindings,
        )
        relationship_summaries.append(dict(result.summary))
        rows = build_pdg_publication_write_rows(
            result,
            cdg_artifact_envelope=PDGCDGArtifactEnvelope(
                fqdn_prefix="physics.pdg.remote_wave.cdg",
                semver="2026.5.5",
                namespace_root="physics",
                namespace_path="pdg/remote_wave/cdg",
                source_package="pdg",
                source_module_path="pdg.remote_derivations",
                source_symbol_prefix="pdg_remote_wave_cdg",
                status="draft",
                visibility_tier="internal",
                description="Remote PDG-derived CDG from Cypher source payload",
                is_latest=True,
            ),
        )
        pdg_rows = _with_stable_relationship_ids(rows.to_insert_rows())
        diagnostics = list(validate_pdg_cdg_publication_graph(pdg_rows))
        errors = [
            row for row in diagnostics if row.get("severity", "error") == "error"
        ]
        if errors:
            raise ValueError(
                f"PDG CDG graph validation failed for {payload.get('derivation_id')}: {errors}"
            )
        graph_diagnostics.extend(dict(row) for row in diagnostics)
        publication_summaries.append(dict(rows.summary))
        additional_rows.append(pdg_rows)

    source_pipeline = run_physics_publication_pipeline(
        source_bundles=[source_bundle],
        table_modes={
            "physics_ingest_snapshots": "upsert",
            "physics_equation_candidates": "upsert",
        },
        dry_run=False,
    )
    insert_rows = merge_publication_insert_rows(
        source_pipeline.write_plan.to_insert_rows(),
        _symbolic_rows_by_table(symbolic_rows.values()),
        *additional_rows,
    )
    insert_rows = _dedupe_insert_rows_by_conflict_key(insert_rows)
    write_plan = build_publication_write_plan(
        insert_rows,
        table_modes=DEFAULT_TABLE_MODES,
    )
    apply_report = _apply_if_requested(args.apply, write_plan)
    out_dir = args.out_dir or _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": args.repo,
        "ref": args.ref,
        "repo_pushed_at": str(repo_metadata.get("pushed_at") or ""),
        "source_version": args.source_version,
        "selected_derivation_ids": list(selected_ids),
        "promoted_derivation_count": len(payloads),
        "snapshot_id": planned_source_bundle["snapshot_row"]["snapshot_id"],
        "equation_count": len(source_bundle_for_snapshot.equations),
        "inference_edge_count": len(source_bundle_for_snapshot.inference_edges),
        "relationship_summaries": relationship_summaries,
        "pdg_publication_summaries": publication_summaries,
        "graph_diagnostic_count": len(graph_diagnostics),
        "table_row_counts": dict(write_plan.audit_summary.planned_row_counts),
        "write_plan_total_row_count": write_plan.audit_summary.total_row_count,
        "apply": apply_report["summary"],
        "github_request_count": len(client.request_log),
        "github_rate_limit_remaining": (
            client.request_log[-1].get("rate_limit_remaining")
            if client.request_log
            else None
        ),
    }
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "repo_request_log.json", client.request_log)
    _write_json(out_dir / "selected_derivation_payloads.json", payloads)
    _write_json(out_dir / "source_bundle.json", source_bundle)
    _write_json(out_dir / "planned_source_bundle.json", planned_source_bundle)
    _write_json(out_dir / "symbolic_rows_by_pdg_node.json", symbolic_rows)
    _write_json(out_dir / "graph_diagnostics.json", graph_diagnostics)
    _write_json(out_dir / "insert_rows_by_table.json", write_plan.to_insert_rows())
    _write_json(out_dir / "write_plan.json", write_plan.to_dict())
    _write_json(out_dir / "apply_report.json", apply_report)
    _write_text(out_dir / "PDG_REMOTE_PROMOTION_REPORT.md", _promotion_markdown(summary, payloads))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if apply_report["summary"]["ok"] else 2


def parse_remote_pdg_cypher(core_texts: Mapping[str, str]) -> PDGRemoteGraph:
    nodes_by_label: dict[str, dict[str, CypherNode]] = defaultdict(dict)
    relations: list[CypherRelation] = []
    file_hashes: dict[str, str] = {}
    for path, text in sorted(core_texts.items()):
        file_hashes[path] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        for match in _NODE_RE.finditer(text):
            node = CypherNode(
                node_id=match.group("id"),
                label=match.group("label"),
                properties=_parse_properties(match.group("properties")),
            )
            nodes_by_label[node.label][node.node_id] = node
        for match in _RELATION_RE.finditer(text):
            relations.append(
                CypherRelation(
                    source_id=match.group("start"),
                    target_id=match.group("end"),
                    source_label=match.group("start_label"),
                    target_label=match.group("end_label"),
                    relation=match.group("rel"),
                    properties=_parse_properties(match.group("props")),
                )
            )
    return PDGRemoteGraph(
        nodes_by_label={label: dict(nodes) for label, nodes in nodes_by_label.items()},
        relations=tuple(relations),
        core_file_sha256=file_hashes,
    )


def build_selected_derivation_payloads(
    graph: PDGRemoteGraph,
    selected_derivation_ids: Iterable[str],
    *,
    repo: str,
    ref: str,
    source_version: str,
    license_expression: str,
    license_evidence_url: str,
) -> list[dict[str, Any]]:
    records = _step_records(graph)
    payloads: list[dict[str, Any]] = []
    for derivation_id in selected_derivation_ids:
        steps = records.get(derivation_id, ())
        if not steps:
            continue
        derivation = graph.nodes_by_label.get("derivation", {}).get(derivation_id)
        expression_ids = sorted(
            {
                expression_id
                for step in steps
                for expression_id in (
                    *step.input_expression_ids,
                    *step.output_expression_ids,
                )
            }
        )
        equations = [
            _expression_node_to_equation(
                graph,
                expression_id,
                derivation_id=derivation_id,
                source_version=source_version,
            )
            for expression_id in expression_ids
            if expression_id in graph.nodes_by_label.get("expression", {})
        ]
        edges = []
        for step in steps:
            rule = graph.nodes_by_label.get("inference_rule", {}).get(step.rule_id)
            rule_label = str((rule.properties if rule else {}).get("name_latex") or step.rule_id)
            feeds = [
                _feed_payload(graph, feed_id)
                for feed_id in step.feed_ids
                if feed_id in graph.nodes_by_label.get("feed", {})
            ]
            for source_id in step.input_expression_ids:
                for target_id in step.output_expression_ids:
                    edges.append(
                        {
                            "id": (
                                f"pdg_remote:{derivation_id}:{step.step_id}:"
                                f"{source_id}:{target_id}"
                            ),
                            "source": source_id,
                            "target": target_id,
                            "rule_id": step.rule_id,
                            "rule": rule_label,
                            "confidence": 0.7,
                            "bindings": {
                                "derivation_id": derivation_id,
                                "step_id": step.step_id,
                                "operation_kind": _operation_kind(rule_label),
                                "feeds": feeds,
                            },
                            "evidence": {
                                "repo": repo,
                                "ref": ref,
                                "source_version": source_version,
                                "pdg_derivation_id": derivation_id,
                                "pdg_step_id": step.step_id,
                                "pdg_inference_rule_id": step.rule_id,
                            },
                        }
                    )
        payloads.append(
            {
                "fixture_id": f"remote_pdg_derivation_{derivation_id}",
                "derivation_id": derivation_id,
                "derivation_label": str(
                    (derivation.properties if derivation else {}).get("name_latex")
                    or derivation_id
                ),
                "source_system": "physics_derivation_graph",
                "source_family": "derivation_graph",
                "source_version": source_version,
                "source_uri": f"https://github.com/{repo}/tree/{ref}",
                "license_expression": license_expression,
                "license_evidence_url": license_evidence_url,
                "equations": equations,
                "inference_edges": edges,
                "metadata": {
                    "repo": repo,
                    "ref": ref,
                    "core_file_sha256": dict(graph.core_file_sha256),
                    "step_count": len(steps),
                    "edge_count": len(edges),
                },
            }
        )
    return payloads


def _step_records(graph: PDGRemoteGraph) -> dict[str, tuple[StepRecord, ...]]:
    step_to_derivation: dict[str, tuple[str, int]] = {}
    step_to_rule: dict[str, str] = {}
    step_inputs: dict[str, list[tuple[int, str]]] = defaultdict(list)
    step_outputs: dict[str, list[tuple[int, str]]] = defaultdict(list)
    step_feeds: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for relation in graph.relations:
        if relation.relation == "HAS_STEP":
            step_to_derivation[relation.target_id] = (
                relation.source_id,
                _sequence_index(relation.properties),
            )
        elif relation.relation == "HAS_INFERENCE_RULE":
            step_to_rule[relation.source_id] = relation.target_id
        elif relation.relation == "HAS_INPUT":
            step_inputs[relation.source_id].append(
                (_sequence_index(relation.properties), relation.target_id)
            )
        elif relation.relation == "HAS_OUTPUT":
            step_outputs[relation.source_id].append(
                (_sequence_index(relation.properties), relation.target_id)
            )
        elif relation.relation == "HAS_FEED":
            step_feeds[relation.source_id].append(
                (_sequence_index(relation.properties), relation.target_id)
            )
    by_derivation: dict[str, list[StepRecord]] = defaultdict(list)
    for step_id, (derivation_id, sequence_index) in step_to_derivation.items():
        by_derivation[derivation_id].append(
            StepRecord(
                step_id=step_id,
                derivation_id=derivation_id,
                sequence_index=sequence_index,
                rule_id=step_to_rule.get(step_id, ""),
                input_expression_ids=tuple(
                    item for _, item in sorted(step_inputs.get(step_id, ()))
                ),
                output_expression_ids=tuple(
                    item for _, item in sorted(step_outputs.get(step_id, ()))
                ),
                feed_ids=tuple(item for _, item in sorted(step_feeds.get(step_id, ()))),
            )
        )
    return {
        derivation_id: tuple(sorted(steps, key=lambda step: step.sequence_index))
        for derivation_id, steps in by_derivation.items()
    }


def _expression_node_to_equation(
    graph: PDGRemoteGraph,
    expression_id: str,
    *,
    derivation_id: str,
    source_version: str,
) -> dict[str, Any]:
    node = graph.nodes_by_label["expression"][expression_id]
    props = dict(node.properties)
    formula = _expression_latex(props) or str(props.get("sympy") or "")
    return {
        "id": expression_id,
        "label": str(props.get("name_latex") or formula or expression_id),
        "description": str(props.get("description_latex") or ""),
        "latex": formula,
        "formula_format": "latex" if formula else "sympy",
        "source_uri": f"pdg://remote/{derivation_id}/expression/{expression_id}",
        "raw_payload": {
            **props,
            "pdg_expression_id": expression_id,
            "pdg_derivation_id": derivation_id,
            "source_version": source_version,
        },
    }


def _feed_payload(graph: PDGRemoteGraph, feed_id: str) -> dict[str, Any]:
    node = graph.nodes_by_label["feed"][feed_id]
    return {
        "feed_id": feed_id,
        "latex": str(node.properties.get("latex") or ""),
        "sympy": str(node.properties.get("sympy") or ""),
    }


def _expression_latex(properties: Mapping[str, Any]) -> str:
    lhs = str(properties.get("latex_lhs") or "").strip()
    rhs = str(properties.get("latex_rhs") or "").strip()
    relation = str(properties.get("latex_relation") or "=").strip() or "="
    condition = str(properties.get("latex_condition") or "").strip()
    if lhs and rhs:
        formula = f"{lhs} {relation} {rhs}"
    else:
        formula = str(properties.get("latex") or "").strip()
    if formula and condition:
        formula = f"{formula} \\quad {condition}"
    return formula


def _equation_symbolic_rows(
    equations: Iterable[Any],
    *,
    candidate_id_by_node_id: Mapping[str, str],
    source_version: str,
    repo: str,
    ref: str,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for equation in equations:
        node_id = str(equation.node_id)
        label = str(equation.label or node_id)
        formula = str(equation.formula or "")
        artifact_id = str(uuid5(_ARTIFACT_NAMESPACE, node_id))
        version_id = str(uuid5(_VERSION_NAMESPACE, node_id))
        expression_id = str(uuid5(_EXPRESSION_NAMESPACE, node_id))
        fqdn = f"physics.pdg.remote_wave.equation.{_fqdn_token(node_id)}"
        content_hash = _sha256_json(
            {
                "source": "physics_derivation_graph",
                "source_version": source_version,
                "repo": repo,
                "ref": ref,
                "node_id": node_id,
                "formula": formula,
                "label": label,
            }
        )
        rows[node_id] = {
            "label": label,
            "artifact": {
                "artifact_id": artifact_id,
                "artifact_kind": "atom",
                "fqdn": fqdn,
                "namespace_root": "physics",
                "namespace_path": "pdg/remote_wave/equations",
                "source_package": "pdg",
                "source_module_path": "pdg.remote_equations",
                "source_symbol": _fqdn_token(node_id),
                "status": "draft",
                "visibility_tier": "internal",
                "description": f"Remote PDG equation node: {label}",
                "source_kind": "generated",
                "is_stochastic": False,
                "is_ffi": False,
                "is_publishable": False,
                "topo_hash": "",
                "top_level_input_arity": 0,
                "top_level_output_arity": 1,
                "leaf_count": 1,
                "verified_leaf_coverage": 0.0,
            },
            "version": {
                "version_id": version_id,
                "artifact_id": artifact_id,
                "content_hash": content_hash,
                "semver": "2026.5.5",
                "is_latest": True,
                "s3_key": "",
                "fingerprint": content_hash,
            },
            "expression": {
                "expression_id": expression_id,
                "artifact_id": artifact_id,
                "version_id": version_id,
                "candidate_id": candidate_id_by_node_id.get(node_id),
                "expression_kind": "equation",
                "expression_role": "primary",
                "raw_formula": formula,
                "raw_formula_format": str(equation.formula_format or "latex"),
                "source_expression_id": node_id,
                "parse_status": "raw_imported",
                "parse_confidence": 0.4 if formula else 0.0,
                "review_status": "needs_human",
                "validation_status": "unknown",
                "mechanism_tags": list(equation.mechanism_tags),
                "behavioral_archetypes": list(equation.behavioral_archetypes),
                "assumptions_json": {"assumptions": list(equation.assumptions)},
                "evidence_json": {
                    "source_system": "physics_derivation_graph",
                    "pdg_expression_id": node_id,
                    "repo": repo,
                    "ref": ref,
                    "source_version": source_version,
                    "label": label,
                },
            },
        }
    return rows


def _symbolic_rows_by_table(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    artifacts: list[dict[str, Any]] = []
    versions: list[dict[str, Any]] = []
    expressions: list[dict[str, Any]] = []
    for row in rows:
        artifacts.append(dict(row["artifact"]))
        versions.append(dict(row["version"]))
        expressions.append(
            {
                key: value
                for key, value in dict(row["expression"]).items()
                if value is not None
            }
        )
    return {
        "artifacts": artifacts,
        "artifact_versions": versions,
        "artifact_symbolic_expressions": expressions,
    }


def _with_stable_relationship_ids(
    rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    copied: dict[str, list[dict[str, Any]]] = {}
    for table, rows in rows_by_table.items():
        copied[table] = [dict(row) for row in rows]
    for row in copied.get("artifact_relationships", []):
        if row.get("relationship_id"):
            continue
        seed = {
            "source_kind": row.get("source_kind"),
            "relationship_kind": row.get("relationship_kind"),
            "source_expression_id": row.get("source_expression_id"),
            "target_expression_id": row.get("target_expression_id"),
            "source_node_id": row.get("source_node_id"),
            "target_node_id": row.get("target_node_id"),
            "inference_rule_id": row.get("inference_rule_id"),
        }
        row["relationship_id"] = str(uuid5(_RELATIONSHIP_NAMESPACE, _canonical_json(seed)))
    return copied


def _dedupe_insert_rows_by_conflict_key(
    rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    conflict_keys = {
        "physics_ingest_snapshots": ("snapshot_id",),
        "physics_equation_candidates": ("candidate_id",),
        "artifacts": ("artifact_id",),
        "artifact_versions": ("version_id",),
        "artifact_symbolic_expressions": ("expression_id",),
        "artifact_relationships": ("relationship_id",),
        "artifact_cdg_nodes": ("version_id", "node_id"),
        "artifact_cdg_edges": (
            "version_id",
            "source_id",
            "target_id",
            "output_name",
            "input_name",
        ),
        "artifact_cdg_bindings": ("version_id", "node_id", "bound_artifact_fqdn"),
    }
    deduped: dict[str, list[dict[str, Any]]] = {}
    for table, rows in rows_by_table.items():
        keys = conflict_keys.get(table, ())
        seen: set[tuple[Any, ...]] = set()
        table_rows: list[dict[str, Any]] = []
        for row in rows:
            copied = dict(row)
            if keys:
                key = tuple(copied.get(column) for column in keys)
                if key in seen:
                    continue
                seen.add(key)
            table_rows.append(copied)
        deduped[table] = table_rows
    return deduped


def _parse_properties(raw: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    occupied: list[tuple[int, int]] = []
    for match in _STRING_FIELD_RE.finditer(raw):
        values[match.group("key")] = _unescape_cypher_string(match.group("value"))
        occupied.append(match.span())
    for match in _NUMERIC_FIELD_RE.finditer(raw):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        raw_value = match.group("value")
        values[match.group("key")] = (
            float(raw_value) if "." in raw_value else int(raw_value)
        )
    return values


def _unique_rows_by_id(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id") or "")
        if row_id and row_id not in unique:
            unique[row_id] = dict(row)
    return list(unique.values())


def _sequence_index(properties: Mapping[str, Any]) -> int:
    try:
        return int(properties.get("sequence_index", 0))
    except (TypeError, ValueError):
        return 0


def _unescape_cypher_string(value: str) -> str:
    return value.replace(r"\"", '"').replace(r"\\", "\\")


def _apply_if_requested(apply_requested: bool, write_plan: Any) -> dict[str, Any]:
    if not apply_requested:
        return {
            "summary": {
                "requested": False,
                "performed": False,
                "ok": True,
                "reason": "apply_not_requested",
            }
        }
    credentials = _supabase_credentials()
    if credentials is None:
        return {
            "summary": {
                "requested": True,
                "performed": False,
                "ok": False,
                "reason": "missing_supabase_credentials",
                "required_env": [
                    "SUPABASE_URL or SCIONA_SUPABASE_URL",
                    (
                        "SUPABASE_SERVICE_ROLE_KEY, SUPABASE_SERVICE_KEY, "
                        "SCIONA_SUPABASE_SERVICE_ROLE_KEY, or "
                        "SCIONA_SUPABASE_SERVICE_KEY"
                    ),
                ],
            }
        }

    from supabase import create_client

    from sciona.physics_ingest.supabase_adapter import apply_publication_supabase_write

    client = create_client(credentials["url"], credentials["key"])
    result = apply_publication_supabase_write(
        client,
        write_plan,
        dry_run=False,
    )
    return {
        "summary": {
            "requested": True,
            "performed": True,
            "ok": not result.has_errors,
            "reason": "applied" if not result.has_errors else "apply_errors",
        },
        "write_result": result.to_dict(),
    }


def _supabase_credentials() -> dict[str, str] | None:
    url = os.environ.get("SUPABASE_URL") or os.environ.get("SCIONA_SUPABASE_URL")
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_SERVICE_KEY")
        or os.environ.get("SCIONA_SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SCIONA_SUPABASE_SERVICE_KEY")
    )
    if not url or not key:
        return None
    return {"url": url, "key": key}


def _default_out_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "output" / f"physics_pdg_remote_wave_{stamp}"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _fqdn_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return token or "equation"


def _promotion_markdown(summary: Mapping[str, Any], payloads: Iterable[Mapping[str, Any]]) -> str:
    lines = [
        "# PDG Remote Promotion Report",
        "",
        f"- Repository: `{summary.get('repo', '')}`",
        f"- Ref: `{summary.get('ref', '')}`",
        f"- Source version: `{summary.get('source_version', '')}`",
        f"- Promoted derivations: `{summary.get('promoted_derivation_count', 0)}`",
        f"- Equations: `{summary.get('equation_count', 0)}`",
        f"- Inference edges: `{summary.get('inference_edge_count', 0)}`",
        f"- GitHub requests: `{summary.get('github_request_count', 0)}`",
        f"- GitHub remaining quota: `{summary.get('github_rate_limit_remaining', '')}`",
        "",
    ]
    for payload in payloads:
        equations = list(payload.get("equations") or [])
        edges = list(payload.get("inference_edges") or [])
        lines.extend(
            [
                f"## {payload.get('derivation_id', '')}",
                "",
                str(payload.get("derivation_label") or ""),
                "",
                f"- equations: `{len(equations)}`",
                f"- edges: `{len(edges)}`",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
