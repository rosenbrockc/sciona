"""FastAPI server for browsing CDGs stored in Memgraph."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
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


@app.get("/api/cdgs/{repo}")
async def get_cdg(repo: str) -> dict[str, Any]:
    """Return full CDG JSON (nodes + edges + metadata) for a repo."""
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
