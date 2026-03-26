"""CDG listing and loading routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter()


@router.get("/api/cdgs")
async def list_cdgs(
    request: Request,
    concept_type: str | None = Query(None, description="Filter by concept type"),
    status: str | None = Query(None, description="Filter by atom status"),
    q: str | None = Query(None, description="Substring search on repo name"),
) -> list[dict[str, Any]]:
    driver = request.app.state.driver
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
        if concept_type and concept_type not in concept_types:
            continue
        if status and status not in statuses:
            continue
        rows.append(
            {
                "repo": rec["repo"],
                "node_count": rec["node_count"],
                "concept_types": concept_types,
                "statuses": statuses,
            }
        )
    return rows


async def _load_cdg(request: Request, repo: str) -> dict[str, Any]:
    driver = request.app.state.driver
    async with driver.session() as session:
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
    nodes = []
    for rec in node_records:
        atom = dict(rec["a"])
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
            "inputs": [
                {
                    "name": dict(ip).get("name", ""),
                    "type_desc": dict(ip).get("type_desc", ""),
                    "constraints": dict(ip).get("constraints", ""),
                }
                for ip in rec["inputs"]
                if ip is not None
            ],
            "outputs": [
                {
                    "name": dict(op).get("name", ""),
                    "type_desc": dict(op).get("type_desc", ""),
                    "constraints": dict(op).get("constraints", ""),
                }
                for op in rec["outputs"]
                if op is not None
            ],
        }
        children = [c for c in rec["children"] if c is not None]
        if children:
            node["children"] = children
        if rec["parent_id"]:
            node["parent_id"] = rec["parent_id"]
        nodes.append(node)
    edges = [
        {
            "source_id": rec["source_id"],
            "target_id": rec["target_id"],
            "output_name": rec["output_name"] or "",
            "input_name": rec["input_name"] or "",
            "source_type": rec["source_type"] or "",
            "target_type": rec["target_type"] or "",
            "requires_glue": bool(rec["requires_glue"]),
        }
        for rec in edge_records
    ]
    return {"nodes": nodes, "edges": edges, "metadata": {"repo": repo}}


@router.get("/api/cdg")
async def get_cdg(request: Request, repo: str = Query(..., description="Full repo path")) -> dict[str, Any]:
    return await _load_cdg(request, repo)


@router.get("/api/cdgs/{repo:path}")
async def get_cdg_by_path(request: Request, repo: str) -> dict[str, Any]:
    return await _load_cdg(request, repo)
