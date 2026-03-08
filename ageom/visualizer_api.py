"""FastAPI server for browsing CDGs stored in Memgraph."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.types import Receive, Scope, Send


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Create Memgraph driver on startup, close on shutdown."""
    from ageom.config import AgeomConfig
    from neo4j import AsyncGraphDatabase

    config = AgeomConfig()
    auth = (config.memgraph_user, config.memgraph_password) if config.memgraph_user else None
    driver = AsyncGraphDatabase.driver(config.memgraph_uri, auth=auth)
    app.state.driver = driver
    yield
    await driver.close()


app = FastAPI(title="AGEO CDG Visualizer", lifespan=_lifespan)


def _merge_runs(
    persisted: list[dict[str, Any]],
    runtime: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in persisted + runtime:
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        current = merged.get(run_id)
        if current is None:
            merged[run_id] = row
            continue
        if float(row.get("last_update_at", 0.0)) >= float(
            current.get("last_update_at", 0.0)
        ):
            merged[run_id] = row
    rows = list(merged.values())
    rows.sort(key=lambda r: float(r.get("last_update_at", 0.0)), reverse=True)
    return rows


def _annotate_hang_signals(
    run: dict[str, Any],
    *,
    stale_seconds: int,
    now: float,
) -> dict[str, Any]:
    stages = run.get("stages", {}) if isinstance(run.get("stages"), dict) else {}
    stale_stages: list[dict[str, Any]] = []
    for stage_name, stage in stages.items():
        if not isinstance(stage, dict):
            continue
        if str(stage.get("status", "")) != "running":
            continue
        hb = float(stage.get("last_heartbeat_at") or 0.0)
        if hb <= 0:
            continue
        age = max(0.0, now - hb)
        if age >= stale_seconds:
            stale_stages.append(
                {
                    "stage": stage_name,
                    "heartbeat_age_sec": age,
                    "message": str(stage.get("message", "")),
                }
            )
    out = dict(run)
    out["stale_stages"] = stale_stages
    out["is_hung"] = len(stale_stages) > 0 and str(run.get("status")) == "running"
    return out


def _extract_dashboard_summaries(run: dict[str, Any]) -> dict[str, Any]:
    """Derive dashboard-friendly summaries from run metadata."""
    metadata = run.get("metadata", {}) if isinstance(run.get("metadata"), dict) else {}
    retrieval = metadata.get("retrieval_policy", {})
    routing = metadata.get("llm_routing", {})
    benchmark = metadata.get("benchmark_validation", {})
    release_validation = metadata.get("release_validation", {})
    shared_context = metadata.get("shared_context", {})
    if not isinstance(retrieval, dict):
        retrieval = {}
    if not isinstance(routing, dict):
        routing = {}
    if not isinstance(benchmark, dict):
        benchmark = {}
    if not isinstance(release_validation, dict):
        release_validation = {}
    if not isinstance(shared_context, dict):
        shared_context = {}
    contexts = shared_context.get("contexts", {})
    if not isinstance(contexts, dict):
        contexts = {}

    def _transport_for_provider(provider: str) -> str:
        lowered = provider.strip().lower()
        if lowered.endswith("_shim"):
            return "persistent_shim"
        if lowered.endswith("_cli"):
            return "legacy_cli"
        if lowered == "llama_cpp":
            return "local_server"
        if lowered in {"anthropic", "codex", "openai"}:
            return "api"
        if not lowered:
            return "--"
        return "other"

    def _routing_line(name: str) -> dict[str, Any]:
        section = routing.get(name, {})
        if not isinstance(section, dict):
            section = {}
        active = section.get("active_overrides", [])
        if not isinstance(active, list):
            active = []
        return {
            "round": name,
            "default": (
                f"{section.get('default_provider', '--')}:{section.get('default_model', '--')}"
            ),
            "active_count": len(active),
            "suppressed_count": len(section.get("suppressed_default_overrides", []) or []),
            "custom_nonbenchmark_count": len(
                section.get("custom_nonbenchmark_overrides", []) or []
            ),
        }

    provider_models: set[str] = set()
    providers: set[str] = set()
    transports: set[str] = set()
    for section in routing.values():
        if not isinstance(section, dict):
            continue
        default_provider = str(section.get("default_provider", "") or "").strip()
        default_model = str(section.get("default_model", "") or "").strip()
        if default_provider:
            providers.add(default_provider)
            transports.add(_transport_for_provider(default_provider))
            provider_models.add(f"{default_provider}:{default_model or '--'}")
        active = section.get("active_overrides", [])
        if not isinstance(active, list):
            continue
        for row in active:
            if not isinstance(row, dict):
                continue
            provider = str(row.get("provider", "") or "").strip()
            model = str(row.get("model", "") or "").strip()
            if not provider:
                continue
            providers.add(provider)
            transports.add(_transport_for_provider(provider))
            provider_models.add(f"{provider}:{model or '--'}")

    out = dict(run)
    out["retrieval_summary"] = {
        "confidence_band": retrieval.get("confidence_band", "--"),
        "skill_index": bool(retrieval.get("skill_index", False)),
        "graph_retrieval": bool(retrieval.get("graph_retrieval", False)),
        "semantic_backend": retrieval.get("semantic_backend", "default"),
        "hunter_mode": retrieval.get("hunter_mode", "--"),
    }
    out["routing_summary"] = {
        "architect": _routing_line("architect"),
        "hunter": _routing_line("hunter"),
    }
    out["provider_complexity"] = {
        "provider_count": len(providers),
        "provider_model_count": len(provider_models),
        "transport_count": len(transports),
        "providers": sorted(providers),
        "provider_models": sorted(provider_models),
        "transports": sorted(transports),
    }
    out["benchmark_summary"] = {
        "prompt_cases": int(benchmark.get("prompt_cases", 0) or 0),
        "prompt_results": int(benchmark.get("prompt_results", 0) or 0),
        "flow_cases": int(benchmark.get("flow_cases", 0) or 0),
        "flow_results": int(benchmark.get("flow_results", 0) or 0),
        "prompt_summary": str(benchmark.get("prompt_summary", "") or ""),
        "prompt_stability_summary": str(
            benchmark.get("prompt_stability_summary", "") or ""
        ),
        "flow_summary": str(benchmark.get("flow_summary", "") or ""),
        "flow_stability_summary": str(
            benchmark.get("flow_stability_summary", "") or ""
        ),
        "flow_avg_prompt_calls": dict(
            benchmark.get("flow_avg_prompt_calls", {}) or {}
        )
        if isinstance(benchmark.get("flow_avg_prompt_calls", {}), dict)
        else {},
        "summary_report": str(benchmark.get("summary_report", "") or ""),
        "manifest": str(release_validation.get("manifest", "") or ""),
        "benchmarks_dir": str(release_validation.get("benchmarks_dir", "") or ""),
        "release_status": str(release_validation.get("status", "") or ""),
    }
    shared_rows: list[dict[str, Any]] = []
    total_searches = 0
    total_hits = 0
    total_puts = 0
    total_injected_blocks = 0
    total_promotions = 0
    total_template_searches = 0
    total_template_hits = 0
    total_template_puts = 0
    total_template_injected = 0
    total_failure_searches = 0
    total_failure_hits = 0
    total_failure_puts = 0
    total_failure_injected = 0
    backends: set[str] = set()
    active_contexts = 0
    for label, row in contexts.items():
        if not isinstance(row, dict):
            continue
        searches = int(row.get("searches_total", 0) or 0)
        hits = int(row.get("search_hits", 0) or 0)
        puts = int(row.get("puts_total", 0) or 0)
        injected_blocks = int(row.get("injected_blocks", 0) or 0)
        promotions = int(row.get("promotions_total", 0) or 0)
        template_searches = int(row.get("template_searches_total", 0) or 0)
        template_hits = int(row.get("template_search_hits", 0) or 0)
        template_puts = int(row.get("template_puts_total", 0) or 0)
        template_injected = int(row.get("template_injected_blocks", 0) or 0)
        failure_searches = int(row.get("failure_searches_total", 0) or 0)
        failure_hits = int(row.get("failure_search_hits", 0) or 0)
        failure_puts = int(row.get("failure_puts_total", 0) or 0)
        failure_injected = int(row.get("failure_injected_blocks", 0) or 0)
        backend = str(row.get("backend", "") or "").strip()
        if backend:
            backends.add(backend)
        if any(
            value > 0
            for value in (
                searches,
                puts,
                injected_blocks,
                promotions,
                template_searches,
                template_puts,
                template_injected,
                failure_searches,
                failure_puts,
                failure_injected,
            )
        ):
            active_contexts += 1
        total_searches += searches
        total_hits += hits
        total_puts += puts
        total_injected_blocks += injected_blocks
        total_promotions += promotions
        total_template_searches += template_searches
        total_template_hits += template_hits
        total_template_puts += template_puts
        total_template_injected += template_injected
        total_failure_searches += failure_searches
        total_failure_hits += failure_hits
        total_failure_puts += failure_puts
        total_failure_injected += failure_injected
        shared_rows.append(
            {
                "label": label,
                "backend": backend or "--",
                "searches": searches,
                "hits": hits,
                "puts": puts,
                "injected_blocks": injected_blocks,
                "promotions": promotions,
                "template_searches": template_searches,
                "template_hits": template_hits,
                "template_puts": template_puts,
                "template_injected_blocks": template_injected,
                "failure_searches": failure_searches,
                "failure_hits": failure_hits,
                "failure_puts": failure_puts,
                "failure_injected_blocks": failure_injected,
            }
        )
    out["shared_context_summary"] = {
        "context_count": len(shared_rows),
        "active_context_count": active_contexts,
        "backends": sorted(backends),
        "total_searches": total_searches,
        "total_hits": total_hits,
        "total_puts": total_puts,
        "total_injected_blocks": total_injected_blocks,
        "total_promotions": total_promotions,
        "total_template_searches": total_template_searches,
        "total_template_hits": total_template_hits,
        "total_template_puts": total_template_puts,
        "total_template_injected_blocks": total_template_injected,
        "total_failure_searches": total_failure_searches,
        "total_failure_hits": total_failure_hits,
        "total_failure_puts": total_failure_puts,
        "total_failure_injected_blocks": total_failure_injected,
        "metrics_path": str(shared_context.get("metrics_path", "") or ""),
        "contexts": sorted(shared_rows, key=lambda row: row["label"]),
    }
    return out


def _decorate_dashboard_run(
    run: dict[str, Any],
    *,
    stale_seconds: int,
    now: float,
) -> dict[str, Any]:
    """Apply all dashboard-facing derived annotations."""
    return _extract_dashboard_summaries(
        _annotate_hang_signals(run, stale_seconds=stale_seconds, now=now)
    )


@app.get("/api/dashboard/runs")
async def list_dashboard_runs(
    limit: int = Query(50, ge=1, le=500),
    state: str = Query(
        "all", description="Filter by state: all|running|completed|failed"
    ),
) -> dict[str, Any]:
    """List telemetry runs with stale/hang annotations."""
    from ageom.config import AgeomConfig
    from ageom.telemetry import list_runtime_runs, load_persisted_runs

    config = AgeomConfig()
    rows = _merge_runs(
        load_persisted_runs(config.telemetry_runs_dir, limit=max(limit * 3, 100)),
        list_runtime_runs(),
    )
    wanted = state.strip().lower()
    if wanted != "all":
        rows = [r for r in rows if str(r.get("status", "")).lower() == wanted]

    now = time.time()
    rows = [
        _decorate_dashboard_run(
            r, stale_seconds=max(5, int(config.telemetry_stale_seconds)), now=now
        )
        for r in rows[:limit]
    ]
    return {"runs": rows, "count": len(rows)}


@app.get("/api/dashboard/runs/{run_id}")
async def get_dashboard_run(run_id: str) -> dict[str, Any]:
    """Get one telemetry run snapshot."""
    from ageom.config import AgeomConfig
    from ageom.telemetry import get_persisted_run, get_runtime_run

    config = AgeomConfig()
    row = get_runtime_run(run_id) or get_persisted_run(config.telemetry_runs_dir, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return _decorate_dashboard_run(
        row,
        stale_seconds=max(5, int(config.telemetry_stale_seconds)),
        now=time.time(),
    )


@app.get("/api/dashboard/latest")
async def get_latest_dashboard_run() -> dict[str, Any]:
    """Get the most recently-updated telemetry run snapshot."""
    payload = await list_dashboard_runs(limit=1, state="all")
    runs = payload.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise HTTPException(status_code=404, detail="No telemetry runs found")
    latest = runs[0]
    if not isinstance(latest, dict):
        raise HTTPException(status_code=404, detail="No telemetry runs found")
    return latest


@app.get("/api/cdgs")
async def list_cdgs(
    concept_type: str | None = Query(None, description="Filter by concept type"),
    status: str | None = Query(None, description="Filter by atom status"),
    q: str | None = Query(None, description="Substring search on repo name"),
) -> list[dict[str, Any]]:
    """List available CDGs with summary stats."""
    driver = app.state.driver

    # Build WHERE clauses
    where_clauses: list[str] = []
    params: dict[str, Any] = {}
    if q:
        where_clauses.append("a.repo CONTAINS $q")
        params["q"] = q

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    cypher = f"""
    MATCH (a:Atom)
    {where_sql}
    WITH a.repo AS repo,
         count(a) AS node_count,
         collect(DISTINCT a.concept_type) AS concept_types,
         collect(DISTINCT a.status) AS statuses
    RETURN repo, node_count, concept_types, statuses
    ORDER BY repo
    """

    async with driver.session() as session:
        result = await session.run(cypher, **params)
        records = [r async for r in result]

    rows = []
    for rec in records:
        concept_types = [ct for ct in rec["concept_types"] if ct]
        statuses = [s for s in rec["statuses"] if s]

        # Apply post-query filters
        if concept_type and concept_type not in concept_types:
            continue
        if status and status not in statuses:
            continue

        rows.append({
            "repo": rec["repo"],
            "node_count": rec["node_count"],
            "concept_types": concept_types,
            "statuses": statuses,
        })

    return rows


async def _load_cdg(repo: str) -> dict[str, Any]:
    """Load full CDG JSON (nodes + edges + metadata) for a repo."""
    driver = app.state.driver

    async with driver.session() as session:
        # Fetch nodes — use explicit parameters dict for Memgraph compatibility
        node_result = await session.run(
            """
            MATCH (a:Atom)
            WHERE a.repo = $repo
            OPTIONAL MATCH (a)-[:HAS_INPUT]->(ip:InputPort)
            OPTIONAL MATCH (a)-[:HAS_OUTPUT]->(op:OutputPort)
            OPTIONAL MATCH (a)-[:PARENT_OF]->(child:Atom)
            OPTIONAL MATCH (parent:Atom)-[:PARENT_OF]->(a)
            RETURN a, collect(DISTINCT ip) AS inputs,
                   collect(DISTINCT op) AS outputs,
                   collect(DISTINCT child.node_id) AS children,
                   parent.node_id AS parent_id
            """,
            parameters={"repo": repo},
        )
        node_records = [r async for r in node_result]

        if not node_records:
            raise HTTPException(status_code=404, detail=f"CDG not found: {repo}")

        # Fetch edges
        edge_result = await session.run(
            """
            MATCH (s:Atom)-[r:DATA_FLOW]->(t:Atom)
            WHERE s.repo = $repo AND t.repo = $repo
            RETURN s.node_id AS source_id, t.node_id AS target_id,
                   r.output_name AS output_name, r.input_name AS input_name,
                   r.source_type AS source_type, r.target_type AS target_type,
                   r.requires_glue AS requires_glue
            """,
            parameters={"repo": repo},
        )
        edge_records = [r async for r in edge_result]

    # Build nodes list
    nodes = []
    for rec in node_records:
        atom = dict(rec["a"])
        # Strip internal props, keep domain props
        node: dict[str, Any] = {
            "node_id": atom.get("node_id", ""),
            "name": atom.get("name", ""),
            "description": atom.get("description", ""),
            "concept_type": atom.get("concept_type", ""),
            "status": atom.get("status", "atomic"),
            "depth": atom.get("depth", 0),
            "type_signature": atom.get("type_signature", ""),
            "is_optional": atom.get("is_optional", False),
            "is_opaque": atom.get("is_opaque", False),
            "is_external": atom.get("is_external", False),
            "parallelizable": atom.get("parallelizable", False),
            "conceptual_summary": atom.get("conceptual_summary", ""),
        }

        # Inputs/outputs
        node["inputs"] = [
            {
                "name": dict(ip).get("name", ""),
                "type_desc": dict(ip).get("type_desc", ""),
                "constraints": dict(ip).get("constraints", ""),
            }
            for ip in rec["inputs"]
            if ip is not None
        ]
        node["outputs"] = [
            {
                "name": dict(op).get("name", ""),
                "type_desc": dict(op).get("type_desc", ""),
                "constraints": dict(op).get("constraints", ""),
            }
            for op in rec["outputs"]
            if op is not None
        ]

        # Children / parent
        children = [c for c in rec["children"] if c is not None]
        if children:
            node["children"] = children
        if rec["parent_id"]:
            node["parent_id"] = rec["parent_id"]

        nodes.append(node)

    # Build edges list
    edges = []
    for rec in edge_records:
        edges.append({
            "source_id": rec["source_id"],
            "target_id": rec["target_id"],
            "output_name": rec["output_name"] or "",
            "input_name": rec["input_name"] or "",
            "source_type": rec["source_type"] or "",
            "target_type": rec["target_type"] or "",
            "requires_glue": bool(rec["requires_glue"]),
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {"repo": repo},
    }


@app.get("/api/cdg")
async def get_cdg(repo: str = Query(..., description="Full repo path")) -> dict[str, Any]:
    """Return full CDG JSON (nodes + edges + metadata) for a repo."""
    return await _load_cdg(repo)


@app.get("/api/cdgs/{repo:path}")
async def get_cdg_by_path(repo: str) -> dict[str, Any]:
    """Backward-compatible path form: /api/cdgs/{repo}."""
    return await _load_cdg(repo)


class IsomorphismQuery(BaseModel):
    repo: str
    node_id: str
    radius: int = 0
    min_jaccard: float = 0.3
    max_results: int = 20
    layers: list[int] = [1, 2, 3]


@app.post("/api/isomorphisms")
async def find_isomorphisms(query: IsomorphismQuery) -> dict[str, Any]:
    """Find similar subgraphs using 3-layer retrieval."""
    driver = app.state.driver

    async with driver.session() as session:
        # 1. Resolve the target node
        node_result = await session.run(
            """
            MATCH (a:Atom)
            WHERE a.repo = $repo AND a.node_id = $node_id
            OPTIONAL MATCH (a)-[:PARENT_OF]->(child:Atom)
            RETURN a, collect(DISTINCT child.concept_type) AS child_types,
                   count(DISTINCT child) AS n_children
            """,
            parameters={"repo": query.repo, "node_id": query.node_id},
        )
        node_rec = await node_result.single()
        if not node_rec:
            raise HTTPException(status_code=404, detail=f"Node not found: {query.repo}/{query.node_id}")

        target = dict(node_rec["a"])
        n_children = node_rec["n_children"]

        # If the node is atomic (no children), walk up to nearest decomposed ancestor
        if n_children == 0 or target.get("status") == "atomic":
            parent_result = await session.run(
                """
                MATCH (child:Atom)
                WHERE child.repo = $repo AND child.node_id = $node_id
                MATCH (parent:Atom)-[:PARENT_OF]->(child)
                OPTIONAL MATCH (parent)-[:PARENT_OF]->(sibling:Atom)
                RETURN parent, collect(DISTINCT sibling.concept_type) AS child_types,
                       count(DISTINCT sibling) AS n_children
                """,
                parameters={"repo": query.repo, "node_id": query.node_id},
            )
            parent_rec = await parent_result.single()
            if parent_rec:
                target = dict(parent_rec["parent"])
                n_children = parent_rec["n_children"]

        # If radius > 0 and we have a parent, walk up further
        if query.radius > 0:
            for _ in range(query.radius):
                up_result = await session.run(
                    """
                    MATCH (child:Atom)
                    WHERE child.repo = $repo AND child.node_id = $node_id
                    MATCH (parent:Atom)-[:PARENT_OF]->(child)
                    OPTIONAL MATCH (parent)-[:PARENT_OF]->(sibling:Atom)
                    RETURN parent, collect(DISTINCT sibling.concept_type) AS child_types,
                           count(DISTINCT sibling) AS n_children
                    """,
                    parameters={"repo": target.get("repo", query.repo), "node_id": target.get("node_id", "")},
                )
                up_rec = await up_result.single()
                if up_rec:
                    target = dict(up_rec["parent"])
                    n_children = up_rec["n_children"]
                else:
                    break

        target_fqn = target.get("fqn", f"{target.get('repo', query.repo)}.{target.get('node_id', '')}")
        target_repo = target.get("repo", query.repo)

        query_node = {
            "fqn": target_fqn,
            "name": target.get("name", ""),
            "concept_type": target.get("concept_type", ""),
            "n_children": n_children,
        }

        # Collect results by fqn, keep highest score
        results_by_fqn: dict[str, dict[str, Any]] = {}

        # Layer 1: topo_hash exact match
        topo_hash = target.get("topo_hash")
        if 1 in query.layers and topo_hash:
            topo_result = await session.run(
                """
                MATCH (parent:Atom:Decomposed)
                WHERE parent.topo_hash = $topo_hash AND parent.repo <> $exclude_repo
                MATCH (parent)-[:PARENT_OF]->(child:Atom)
                WITH parent, collect(DISTINCT child.concept_type) AS child_types,
                     count(DISTINCT child) AS n_children
                RETURN parent.fqn AS fqn, parent.repo AS repo, parent.name AS name,
                       parent.concept_type AS concept_type, parent.topo_hash AS topo_hash,
                       n_children, child_types
                LIMIT $limit
                """,
                parameters={"topo_hash": topo_hash, "exclude_repo": target_repo, "limit": query.max_results},
            )
            async for rec in topo_result:
                fqn = rec["fqn"]
                results_by_fqn[fqn] = {
                    "fqn": fqn,
                    "repo": rec["repo"],
                    "name": rec["name"],
                    "concept_type": rec["concept_type"],
                    "topo_hash": rec["topo_hash"],
                    "n_children": rec["n_children"],
                    "score": 1.0,
                    "jaccard_score": None,
                    "layer": 1,
                    "children_summary": rec["child_types"],
                }

        # Layer 2: structural match (concept_type + port arity ±1)
        if 2 in query.layers:
            struct_result = await session.run(
                """
                MATCH (parent:Atom:Decomposed)
                WHERE parent.concept_type = $concept_type
                  AND parent.repo <> $exclude_repo
                  AND abs(parent.n_inputs - $n_inputs) <= 1
                  AND abs(parent.n_outputs - $n_outputs) <= 1
                MATCH (parent)-[:PARENT_OF]->(child:Atom)
                WITH parent, collect(DISTINCT child.concept_type) AS child_types,
                     count(DISTINCT child) AS n_children
                WHERE n_children >= 2
                RETURN parent.fqn AS fqn, parent.repo AS repo, parent.name AS name,
                       parent.concept_type AS concept_type, parent.topo_hash AS topo_hash,
                       n_children, child_types
                ORDER BY n_children DESC
                LIMIT $limit
                """,
                parameters={
                    "concept_type": target.get("concept_type", ""),
                    "n_inputs": target.get("n_inputs", 0) or 0,
                    "n_outputs": target.get("n_outputs", 0) or 0,
                    "exclude_repo": target_repo,
                    "limit": query.max_results,
                },
            )
            async for rec in struct_result:
                fqn = rec["fqn"]
                # Compute IO match score
                n_in_diff = abs((target.get("n_inputs", 0) or 0) - (rec["n_children"] or 0))
                io_match = 1.0 if n_in_diff == 0 else 0.8
                score = 0.7 * io_match
                if fqn not in results_by_fqn or results_by_fqn[fqn]["score"] < score:
                    results_by_fqn[fqn] = {
                        "fqn": fqn,
                        "repo": rec["repo"],
                        "name": rec["name"],
                        "concept_type": rec["concept_type"],
                        "topo_hash": rec["topo_hash"],
                        "n_children": rec["n_children"],
                        "score": score,
                        "jaccard_score": None,
                        "layer": 2,
                        "children_summary": rec["child_types"],
                    }

        # Layer 3: Jaccard neighborhood
        if 3 in query.layers:
            jaccard_result = await session.run(
                """
                MATCH (query:Atom)
                WHERE query.fqn = $fqn
                MATCH (query)-[:PARENT_OF]->(qc:Atom)
                WITH query, collect(qc) AS query_children
                WHERE size(query_children) > 0
                MATCH (candidate:Atom:Decomposed)-[:PARENT_OF]->(cc:Atom)
                WHERE candidate.repo <> $exclude_repo AND candidate.fqn <> $fqn
                WITH query, query_children, candidate, collect(cc) AS cand_children
                WHERE size(cand_children) > 0
                WITH candidate,
                     [qc IN query_children | qc.concept_type] AS q_types,
                     [cc IN cand_children | cc.concept_type] AS c_types
                WITH candidate,
                     toFloat(size([x IN q_types WHERE x IN c_types])) /
                     toFloat(size(q_types + [y IN c_types WHERE NOT y IN q_types])) AS jaccard_score
                WHERE jaccard_score > $min_jaccard
                WITH candidate, jaccard_score
                ORDER BY jaccard_score DESC
                LIMIT $limit
                MATCH (candidate)-[:PARENT_OF]->(child:Atom)
                WITH candidate, jaccard_score,
                     collect(DISTINCT child.concept_type) AS child_types,
                     count(DISTINCT child) AS n_children
                RETURN candidate.fqn AS fqn, candidate.repo AS repo,
                       candidate.name AS name, candidate.concept_type AS concept_type,
                       candidate.topo_hash AS topo_hash,
                       n_children, child_types, jaccard_score
                """,
                parameters={
                    "fqn": target_fqn,
                    "exclude_repo": target_repo,
                    "min_jaccard": query.min_jaccard,
                    "limit": query.max_results,
                },
            )
            async for rec in jaccard_result:
                fqn = rec["fqn"]
                j_score = rec["jaccard_score"]
                score = j_score  # Jaccard score IS the score for layer 3
                existing = results_by_fqn.get(fqn)
                if existing:
                    existing["jaccard_score"] = j_score
                    if score > existing["score"]:
                        existing["score"] = score
                        existing["layer"] = 3
                else:
                    results_by_fqn[fqn] = {
                        "fqn": fqn,
                        "repo": rec["repo"],
                        "name": rec["name"],
                        "concept_type": rec["concept_type"],
                        "topo_hash": rec["topo_hash"],
                        "n_children": rec["n_children"],
                        "score": score,
                        "jaccard_score": j_score,
                        "layer": 3,
                        "children_summary": rec["child_types"],
                    }

    # Sort by score descending, cap at max_results
    results = sorted(results_by_fqn.values(), key=lambda r: r["score"], reverse=True)
    results = results[: query.max_results]

    return {"query_node": query_node, "results": results}


class NoCacheStaticFiles(StaticFiles):
    """StaticFiles that sends no-cache headers so dev reloads always get fresh files."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        original_send = send

        async def send_with_no_cache(message: dict) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"cache-control", b"no-cache, no-store, must-revalidate"))
                message["headers"] = headers
            await original_send(message)

        await super().__call__(scope, receive, send_with_no_cache)


# Mount static files last so API routes take priority
_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.exists():
    app.mount("/", NoCacheStaticFiles(directory=str(_static_dir), html=True), name="static")
