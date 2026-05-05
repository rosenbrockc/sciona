#!/usr/bin/env python
"""Inventory remote Physics Derivation Graph resources for staged ingestion."""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_REPO = "allofphysicsgraph/ui_v8_website_flask_neo4j"
DEFAULT_REF = "gh-pages"
DEFAULT_USER_AGENT = "sciona-physics-ingest-pdg-inventory/0.1"
CORE_CYPHER_PATHS = (
    "conversion_of_data_formats/deriv.cypher",
    "conversion_of_data_formats/infrules.cypher",
    "conversion_of_data_formats/operators.cypher",
    "conversion_of_data_formats/steps.cypher",
    "conversion_of_data_formats/expr_and_feed.cypher",
    "conversion_of_data_formats/symbols.cypher",
)
STRUCTURAL_CORE_PATHS = {
    "conversion_of_data_formats/deriv.cypher",
    "conversion_of_data_formats/infrules.cypher",
    "conversion_of_data_formats/operators.cypher",
    "conversion_of_data_formats/steps.cypher",
    "conversion_of_data_formats/expr_and_feed.cypher",
    "conversion_of_data_formats/symbols.cypher",
    "conversion_of_data_formats/symbols_vector_manual.cypher",
}
_RELATION_RE = re.compile(
    r'UNWIND \[\{start: \{id:"(?P<start>[^"]+)"\}, '
    r'end: \{id:"(?P<end>[^"]+)"\}, properties:\{(?P<props>.*?)\}\}\] AS row\s+'
    r"MATCH \(start:(?P<start_label>\w+).*?\s+"
    r"MATCH \(end:(?P<end_label>\w+).*?\s+"
    r"CREATE \(start\)-\[r:(?P<rel>\w+)\]->\(end\)",
    re.DOTALL,
)
_NODE_CREATE_RE = re.compile(r"CREATE \(n:(?P<label>\w+)\{id: row\.id\}\)")
_ID_PROPERTIES_RE = re.compile(
    r'UNWIND \[\{id:"(?P<id>[^"]+)",\s+properties:\{(?P<properties>.*?)\}\}\] AS row\s+'
    r"CREATE \(n:(?P<label>\w+)\{id: row\.id\}\)",
    re.DOTALL,
)
_NAME_LATEX_RE = re.compile(r'name_latex:"(?P<name>(?:[^"\\]|\\.)*)"')


@dataclass
class RateLimitedGitHubClient:
    """Tiny GitHub API client with explicit request budget and spacing."""

    user_agent: str
    min_delay_seconds: float = 2.0
    max_requests: int = 8
    rate_limit_reserve: int = 10
    token: str = ""
    request_log: list[dict[str, Any]] = field(default_factory=list)
    _last_request_at: float | None = None

    def get_json(self, url: str) -> Any:
        if len(self.request_log) >= self.max_requests:
            raise RuntimeError(
                f"request budget exhausted before {url}; max_requests={self.max_requests}"
            )
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            sleep_for = self.min_delay_seconds - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": self.user_agent,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        started = time.monotonic()
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
                self._record_request(url, response.status, response.headers, started)
                return payload
        except HTTPError as exc:
            self._record_request(url, exc.code, exc.headers, started)
            raise
        finally:
            self._last_request_at = time.monotonic()

    def should_stop_for_rate_reserve(self) -> bool:
        if not self.request_log:
            return False
        remaining = self.request_log[-1].get("rate_limit_remaining")
        return isinstance(remaining, int) and remaining <= self.rate_limit_reserve

    def _record_request(
        self,
        url: str,
        status: int,
        headers: Mapping[str, Any],
        started: float,
    ) -> None:
        self.request_log.append(
            {
                "url": url,
                "status": status,
                "duration_seconds": round(time.monotonic() - started, 3),
                "rate_limit_limit": _optional_int(headers.get("X-RateLimit-Limit")),
                "rate_limit_remaining": _optional_int(
                    headers.get("X-RateLimit-Remaining")
                ),
                "rate_limit_used": _optional_int(headers.get("X-RateLimit-Used")),
                "rate_limit_reset": _optional_int(headers.get("X-RateLimit-Reset")),
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory the remote Physics Derivation Graph repository and classify "
            "resources into Sciona ingestion waves."
        )
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub owner/repo.")
    parser.add_argument("--ref", default=DEFAULT_REF, help="Git ref to inventory.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for inventory artifacts.",
    )
    parser.add_argument(
        "--fetch-core-files",
        action="store_true",
        help="Fetch core Cypher files for derivation-level inventory.",
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
        default=8,
        help="Hard request budget for this run.",
    )
    parser.add_argument(
        "--rate-limit-reserve",
        type=int,
        default=10,
        help="Stop optional fetches when remaining GitHub quota reaches this value.",
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
    parser.add_argument(
        "--license-expression",
        default="",
        help="Override license expression when GitHub repo metadata is inconclusive.",
    )
    parser.add_argument(
        "--license-evidence-url",
        default="https://derivationmap.net/developer_documentation",
        help="Source URL supporting the explicit license expression.",
    )
    args = parser.parse_args()

    client = RateLimitedGitHubClient(
        user_agent=args.user_agent,
        min_delay_seconds=max(args.min_delay_seconds, 0.0),
        max_requests=max(args.max_requests, 1),
        rate_limit_reserve=max(args.rate_limit_reserve, 0),
        token=str(os.environ.get(args.github_token_env) or ""),
    )
    repo_metadata = client.get_json(_github_api_url(args.repo))
    tree_payload = client.get_json(_github_tree_url(args.repo, args.ref))
    core_texts: dict[str, str] = {}
    if args.fetch_core_files:
        for path in CORE_CYPHER_PATHS:
            if client.should_stop_for_rate_reserve():
                break
            if not _tree_has_path(tree_payload, path):
                continue
            content = client.get_json(_github_contents_url(args.repo, path, args.ref))
            core_texts[path] = _decode_github_content(content)

    inventory = build_pdg_remote_inventory(
        repo=args.repo,
        ref=args.ref,
        repo_metadata=repo_metadata,
        tree_payload=tree_payload,
        core_texts=core_texts,
        request_log=client.request_log,
        license_expression_override=args.license_expression,
        license_evidence_url=args.license_evidence_url,
    )
    out_dir = args.out_dir or _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "pdg_remote_inventory_manifest.json", inventory)
    _write_json(out_dir / "summary.json", inventory["summary"])
    _write_json(out_dir / "repository_metadata.json", inventory["repository"])
    _write_json(out_dir / "tree_inventory.json", inventory["tree_inventory"])
    _write_json(out_dir / "core_file_analysis.json", inventory["core_file_analysis"])
    _write_json(out_dir / "pdg_derivations_inventory.json", inventory["derivations"])
    _write_json(out_dir / "ingestion_waves.json", inventory["ingestion_waves"])
    _write_json(out_dir / "rate_limit_report.json", inventory["rate_limit_report"])
    _write_text(out_dir / "PDG_REMOTE_INGESTION_WAVES.md", _waves_markdown(inventory))
    print(json.dumps(inventory["summary"], indent=2, sort_keys=True))
    return 0


def build_pdg_remote_inventory(
    *,
    repo: str,
    ref: str,
    repo_metadata: Mapping[str, Any],
    tree_payload: Mapping[str, Any],
    core_texts: Mapping[str, str] | None = None,
    request_log: Iterable[Mapping[str, Any]] = (),
    license_expression_override: str = "",
    license_evidence_url: str = "",
) -> dict[str, Any]:
    tree_entries = [
        dict(item)
        for item in tree_payload.get("tree", [])
        if isinstance(item, Mapping)
    ]
    files = [entry for entry in tree_entries if entry.get("type") == "blob"]
    tree_inventory = _build_tree_inventory(files, tree_payload=tree_payload)
    core_analysis = analyze_core_files(core_texts or {})
    derivations = _build_derivation_units(core_analysis)
    ingestion_waves = _build_ingestion_waves(files, derivations, core_analysis)
    request_rows = [dict(row) for row in request_log]
    license_expression = license_expression_override or _license_expression(repo_metadata)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": repo,
        "ref": ref,
        "default_branch": str(repo_metadata.get("default_branch") or ""),
        "pushed_at": str(repo_metadata.get("pushed_at") or ""),
        "license": license_expression,
        "license_evidence_url": license_evidence_url,
        "tree_truncated": bool(tree_payload.get("truncated")),
        "tree_entry_count": len(tree_entries),
        "file_count": len(files),
        "core_file_count": len(core_texts or {}),
        "derivation_count": len(derivations),
        "wave_counts": {
            wave["wave_id"]: len(wave["items"])
            for wave in ingestion_waves
        },
        "github_request_count": len(request_rows),
        "github_rate_limit_remaining": (
            request_rows[-1].get("rate_limit_remaining") if request_rows else None
        ),
        "selection_policy": (
            "Preserve every license-allowed PDG resource as raw source evidence; "
            "promote first the derivations with parseable graph structure, known "
            "operation kinds, and complete input/output expression links."
        ),
    }
    return {
        "manifest_version": "pdg-remote-inventory.v1",
        "summary": summary,
        "repository": {
            "full_name": str(repo_metadata.get("full_name") or repo),
            "html_url": str(repo_metadata.get("html_url") or ""),
            "default_branch": str(repo_metadata.get("default_branch") or ""),
            "pushed_at": str(repo_metadata.get("pushed_at") or ""),
            "license": license_expression,
            "license_evidence_url": license_evidence_url,
            "ref": ref,
        },
        "tree_inventory": tree_inventory,
        "core_file_analysis": core_analysis,
        "derivations": derivations,
        "ingestion_waves": ingestion_waves,
        "rate_limit_report": {
            "policy": {
                "uses_github_api": True,
                "single_tree_request": True,
                "optional_core_file_fetches": sorted((core_texts or {}).keys()),
                "no_recursive_file_download_by_default": True,
            },
            "requests": request_rows,
        },
    }


def analyze_core_files(core_texts: Mapping[str, str]) -> dict[str, Any]:
    file_summaries: dict[str, Any] = {}
    derivations: dict[str, dict[str, Any]] = {}
    inference_rules: dict[str, dict[str, Any]] = {}
    step_to_derivation: dict[str, str] = {}
    step_to_rule: dict[str, str] = {}
    step_inputs: dict[str, set[str]] = defaultdict(set)
    step_outputs: dict[str, set[str]] = defaultdict(set)
    step_feeds: dict[str, set[str]] = defaultdict(set)
    relation_counts: Counter[str] = Counter()
    node_counts: Counter[str] = Counter()

    for path, text in sorted(core_texts.items()):
        nodes = Counter(match.group("label") for match in _NODE_CREATE_RE.finditer(text))
        relations = Counter(match.group("rel") for match in _RELATION_RE.finditer(text))
        node_counts.update(nodes)
        relation_counts.update(relations)
        file_summaries[path] = {
            "size_bytes": len(text.encode("utf-8")),
            "node_counts": dict(sorted(nodes.items())),
            "relationship_counts": dict(sorted(relations.items())),
            "role": _core_file_role(path),
        }
        for row in _ID_PROPERTIES_RE.finditer(text):
            label = row.group("label")
            row_id = row.group("id")
            name = _extract_name_latex(row.group("properties"))
            if label == "derivation":
                derivations[row_id] = {"derivation_id": row_id, "name_latex": name}
            elif label == "inference_rule":
                inference_rules[row_id] = {
                    "inference_rule_id": row_id,
                    "name_latex": name,
                    "operation_kind": _operation_kind(name),
                }
        for relation in _RELATION_RE.finditer(text):
            start = relation.group("start")
            end = relation.group("end")
            rel = relation.group("rel")
            if rel == "HAS_STEP":
                step_to_derivation[end] = start
            elif rel == "HAS_INFERENCE_RULE":
                step_to_rule[start] = end
            elif rel == "HAS_INPUT":
                step_inputs[start].add(end)
            elif rel == "HAS_OUTPUT":
                step_outputs[start].add(end)
            elif rel == "HAS_FEED":
                step_feeds[start].add(end)

    return {
        "files": file_summaries,
        "node_counts": dict(sorted(node_counts.items())),
        "relationship_counts": dict(sorted(relation_counts.items())),
        "derivations": sorted(derivations.values(), key=lambda row: row["derivation_id"]),
        "inference_rules": sorted(
            inference_rules.values(), key=lambda row: row["inference_rule_id"]
        ),
        "step_graph": _step_graph_summary(
            derivations=derivations,
            inference_rules=inference_rules,
            step_to_derivation=step_to_derivation,
            step_to_rule=step_to_rule,
            step_inputs=step_inputs,
            step_outputs=step_outputs,
            step_feeds=step_feeds,
        ),
    }


def _build_tree_inventory(
    files: Iterable[Mapping[str, Any]],
    *,
    tree_payload: Mapping[str, Any],
) -> dict[str, Any]:
    files_list = list(files)
    by_extension: Counter[str] = Counter()
    by_top_level: Counter[str] = Counter()
    by_role: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    for item in files_list:
        path = str(item.get("path") or "")
        suffix = Path(path).suffix.lower() or "<none>"
        top_level = path.split("/", 1)[0] if path else "<root>"
        role = _path_role(path)
        by_extension[suffix] += 1
        by_top_level[top_level] += 1
        by_role[role] += 1
        if role != "deferred_support":
            candidates.append(
                {
                    "path": path,
                    "sha": str(item.get("sha") or ""),
                    "size": int(item.get("size") or 0),
                    "role": role,
                    "recommended_raw_ingest": role in {
                        "core_graph_payload",
                        "schema_or_parser_support",
                        "documentation_or_provenance",
                    },
                    "recommended_promotion_scope": _promotion_scope(path, role),
                }
            )
    return {
        "truncated": bool(tree_payload.get("truncated")),
        "file_count": len(files_list),
        "by_extension": dict(sorted(by_extension.items())),
        "by_top_level": dict(sorted(by_top_level.items())),
        "by_role": dict(sorted(by_role.items())),
        "candidate_resources": sorted(candidates, key=lambda row: row["path"]),
    }


def _step_graph_summary(
    *,
    derivations: Mapping[str, Mapping[str, Any]],
    inference_rules: Mapping[str, Mapping[str, Any]],
    step_to_derivation: Mapping[str, str],
    step_to_rule: Mapping[str, str],
    step_inputs: Mapping[str, set[str]],
    step_outputs: Mapping[str, set[str]],
    step_feeds: Mapping[str, set[str]],
) -> dict[str, Any]:
    per_derivation: dict[str, dict[str, Any]] = {
        derivation_id: {
            "derivation_id": derivation_id,
            "name_latex": str(derivation.get("name_latex") or ""),
            "step_ids": [],
            "inference_rule_ids": [],
            "input_expression_ids": set(),
            "output_expression_ids": set(),
            "feed_ids": set(),
            "operation_kind_counts": Counter(),
        }
        for derivation_id, derivation in derivations.items()
    }
    for step_id, derivation_id in step_to_derivation.items():
        row = per_derivation.setdefault(
            derivation_id,
            {
                "derivation_id": derivation_id,
                "name_latex": "",
                "step_ids": [],
                "inference_rule_ids": [],
                "input_expression_ids": set(),
                "output_expression_ids": set(),
                "feed_ids": set(),
                "operation_kind_counts": Counter(),
            },
        )
        row["step_ids"].append(step_id)
        rule_id = step_to_rule.get(step_id, "")
        if rule_id:
            row["inference_rule_ids"].append(rule_id)
            operation = str(
                inference_rules.get(rule_id, {}).get("operation_kind") or "unknown"
            )
            row["operation_kind_counts"][operation] += 1
        row["input_expression_ids"].update(step_inputs.get(step_id, set()))
        row["output_expression_ids"].update(step_outputs.get(step_id, set()))
        row["feed_ids"].update(step_feeds.get(step_id, set()))

    return {
        "derivations": [
            {
                "derivation_id": row["derivation_id"],
                "name_latex": row["name_latex"],
                "step_count": len(set(row["step_ids"])),
                "inference_rule_ids": sorted(set(row["inference_rule_ids"])),
                "input_expression_count": len(row["input_expression_ids"]),
                "output_expression_count": len(row["output_expression_ids"]),
                "feed_count": len(row["feed_ids"]),
                "operation_kind_counts": dict(sorted(row["operation_kind_counts"].items())),
            }
            for row in sorted(
                per_derivation.values(), key=lambda value: str(value["derivation_id"])
            )
        ]
    }


def _build_derivation_units(core_analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in core_analysis.get("step_graph", {}).get("derivations", []):
        if not isinstance(row, Mapping):
            continue
        operation_counts = {
            str(key): int(value)
            for key, value in dict(row.get("operation_kind_counts") or {}).items()
        }
        known_operation_count = sum(
            count for op, count in operation_counts.items() if op != "unknown"
        )
        step_count = int(row.get("step_count") or 0)
        input_count = int(row.get("input_expression_count") or 0)
        output_count = int(row.get("output_expression_count") or 0)
        readiness_score = (
            (2 if step_count else 0)
            + (2 if input_count and output_count else 0)
            + (2 if known_operation_count else 0)
            + (1 if str(row.get("name_latex") or "") else 0)
        )
        rows.append(
            {
                "derivation_id": str(row.get("derivation_id") or ""),
                "name_latex": str(row.get("name_latex") or ""),
                "step_count": step_count,
                "input_expression_count": input_count,
                "output_expression_count": output_count,
                "feed_count": int(row.get("feed_count") or 0),
                "inference_rule_ids": list(row.get("inference_rule_ids") or []),
                "operation_kind_counts": operation_counts,
                "known_operation_count": known_operation_count,
                "readiness_score": readiness_score,
                "recommended_wave": _recommended_derivation_wave(
                    step_count=step_count,
                    input_count=input_count,
                    output_count=output_count,
                    known_operation_count=known_operation_count,
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (-int(row["readiness_score"]), str(row["derivation_id"])),
    )


def _build_ingestion_waves(
    files: Iterable[Mapping[str, Any]],
    derivations: Iterable[Mapping[str, Any]],
    core_analysis: Mapping[str, Any],
) -> list[dict[str, Any]]:
    file_rows = [
        {
            "path": str(item.get("path") or ""),
            "role": _path_role(str(item.get("path") or "")),
            "size": int(item.get("size") or 0),
        }
        for item in files
    ]
    raw_items = [
        row for row in file_rows if row["role"] in {
            "core_graph_payload",
            "schema_or_parser_support",
            "documentation_or_provenance",
        }
    ]
    promotion = [
        dict(row) for row in derivations if row.get("recommended_wave") == "wave_1_cdg_candidates"
    ]
    relationship = [
        dict(row)
        for row in derivations
        if row.get("recommended_wave") == "wave_2_relationship_candidates"
    ]
    metadata = [
        dict(row)
        for row in derivations
        if row.get("recommended_wave") == "wave_3_metadata_only"
    ]
    deferred = [row for row in file_rows if row["role"] == "deferred_support"]
    return [
        {
            "wave_id": "wave_0_raw_repository_snapshot",
            "goal": "Preserve immutable repo tree, provenance, schemas, and core graph files.",
            "items": raw_items,
        },
        {
            "wave_id": "wave_1_cdg_candidates",
            "goal": "Promote derivations with steps, input/output expressions, and known operation kinds.",
            "items": promotion,
        },
        {
            "wave_id": "wave_2_relationship_candidates",
            "goal": "Promote parseable derivation relationships that need operation or expression cleanup before CDGs.",
            "items": relationship,
        },
        {
            "wave_id": "wave_3_metadata_only",
            "goal": "Retain derivation records that are not yet graph-complete.",
            "items": metadata,
        },
        {
            "wave_id": "wave_4_deferred_support_assets",
            "goal": "Defer web app, notebooks, tests, Docker, and generated support files unless they carry provenance value.",
            "items": deferred,
        },
        {
            "wave_id": "wave_5_rule_and_symbol_normalization",
            "goal": "Normalize inference rules, operators, symbols, dimensions, and aliases for later CDG compilation.",
            "items": [
                {
                    "core_node_counts": dict(core_analysis.get("node_counts") or {}),
                    "core_relationship_counts": dict(
                        core_analysis.get("relationship_counts") or {}
                    ),
                }
            ],
        },
    ]


def _path_role(path: str) -> str:
    if path in STRUCTURAL_CORE_PATHS:
        return "core_graph_payload"
    if path.endswith(".cypher") and path.startswith("conversion_of_data_formats/"):
        return "core_graph_payload"
    if "schema" in path and path.endswith(".json"):
        return "schema_or_parser_support"
    if path.endswith((".md", ".rst", ".gv", ".dot", ".log")):
        return "documentation_or_provenance"
    if path.startswith(("webserver_for_pdg/library/", "conversion_of_data_formats/")):
        return "schema_or_parser_support"
    return "deferred_support"


def _promotion_scope(path: str, role: str) -> str:
    if role == "core_graph_payload":
        return "raw_plus_structural_parse"
    if role == "schema_or_parser_support":
        return "adapter_support"
    if role == "documentation_or_provenance":
        return "provenance_context"
    return "deferred"


def _core_file_role(path: str) -> str:
    name = Path(path).name
    return {
        "deriv.cypher": "derivation_metadata",
        "infrules.cypher": "inference_rule_catalog",
        "operators.cypher": "operator_catalog",
        "steps.cypher": "derivation_step_edges",
        "expr_and_feed.cypher": "expression_and_feed_catalog",
        "symbols.cypher": "symbol_catalog",
    }.get(name, "core_graph_payload")


def _operation_kind(name: str) -> str:
    text = name.lower()
    patterns = (
        ("nondimensionalize", ("dimensionless", "non-dimensional", "nondimensional")),
        ("differentiate", ("differentiat", "derivative")),
        ("integrate", ("integrat",)),
        ("substitute", ("substitut", "replace")),
        ("solve", ("solve", "isolate")),
        ("limit", ("limit",)),
        ("simplify", ("simplif", "cancel", "collect")),
        ("expand", ("expand", "foil")),
        ("add", ("add ", "addition")),
        ("subtract", ("subtract", "minus")),
        ("multiply", ("multiply", "product")),
        ("divide", ("divide", "division")),
        ("apply_definition", ("definition", "declare")),
        ("separate_variables", ("separation of variables", "separate variable")),
        ("approximate", ("approx",)),
        ("assume", ("assume", "assumption")),
    )
    for operation, needles in patterns:
        if any(needle in text for needle in needles):
            return operation
    return "unknown"


def _recommended_derivation_wave(
    *,
    step_count: int,
    input_count: int,
    output_count: int,
    known_operation_count: int,
) -> str:
    if step_count and input_count and output_count and known_operation_count:
        return "wave_1_cdg_candidates"
    if step_count and (input_count or output_count):
        return "wave_2_relationship_candidates"
    return "wave_3_metadata_only"


def _extract_name_latex(properties: str) -> str:
    match = _NAME_LATEX_RE.search(properties)
    if match is None:
        return ""
    return _unescape_cypher_string(match.group("name"))


def _unescape_cypher_string(value: str) -> str:
    return value.replace(r"\"", '"').replace(r"\\", "\\")


def _license_expression(repo_metadata: Mapping[str, Any]) -> str:
    license_row = repo_metadata.get("license")
    if isinstance(license_row, Mapping):
        spdx = str(license_row.get("spdx_id") or "")
        if spdx and spdx != "NOASSERTION":
            return spdx
        name = str(license_row.get("name") or "")
        if name and name != "Other":
            return name
    return "upstream-license-review-required"


def _tree_has_path(tree_payload: Mapping[str, Any], path: str) -> bool:
    return any(
        isinstance(item, Mapping)
        and item.get("type") == "blob"
        and item.get("path") == path
        for item in tree_payload.get("tree", [])
    )


def _decode_github_content(payload: Mapping[str, Any]) -> str:
    encoding = str(payload.get("encoding") or "")
    content = str(payload.get("content") or "")
    if encoding != "base64":
        raise ValueError(f"unsupported GitHub content encoding: {encoding}")
    return base64.b64decode(content).decode("utf-8", errors="replace")


def _github_api_url(repo: str) -> str:
    return f"https://api.github.com/repos/{repo}"


def _github_tree_url(repo: str, ref: str) -> str:
    return f"https://api.github.com/repos/{repo}/git/trees/{quote(ref, safe='')}?recursive=1"


def _github_contents_url(repo: str, path: str, ref: str) -> str:
    return (
        f"https://api.github.com/repos/{repo}/contents/"
        f"{quote(path, safe='/')}?ref={quote(ref, safe='')}"
    )


def _default_out_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "output" / f"physics_pdg_remote_inventory_{stamp}"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _waves_markdown(inventory: Mapping[str, Any]) -> str:
    summary = dict(inventory.get("summary") or {})
    lines = [
        "# PDG Remote Ingestion Waves",
        "",
        f"- Repository: `{summary.get('repo', '')}`",
        f"- Ref: `{summary.get('ref', '')}`",
        f"- Generated: `{summary.get('generated_at', '')}`",
        f"- Files inventoried: `{summary.get('file_count', 0)}`",
        f"- Derivations inventoried: `{summary.get('derivation_count', 0)}`",
        f"- GitHub requests: `{summary.get('github_request_count', 0)}`",
        "",
        "Policy: preserve every license-allowed PDG resource as raw source evidence; "
        "promote first derivations with parseable graph structure, known operation "
        "kinds, and complete input/output expression links.",
        "",
    ]
    for wave in inventory.get("ingestion_waves", []):
        if not isinstance(wave, Mapping):
            continue
        items = list(wave.get("items") or [])
        lines.extend(
            [
                f"## {wave.get('wave_id', '')}",
                "",
                str(wave.get("goal") or ""),
                "",
                f"Item count: `{len(items)}`",
                "",
            ]
        )
        for item in items[:20]:
            if isinstance(item, Mapping):
                label = (
                    item.get("path")
                    or item.get("name_latex")
                    or item.get("derivation_id")
                    or "core rule/symbol normalization summary"
                )
                detail = item.get("recommended_wave") or item.get("role") or ""
                lines.append(f"- `{label}` {detail}".rstrip())
        if len(items) > 20:
            lines.append(f"- ... {len(items) - 20} more")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
