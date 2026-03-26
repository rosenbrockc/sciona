"""Isomorphism query routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class IsomorphismQuery(BaseModel):
    repo: str
    node_id: str
    radius: int = 0
    min_jaccard: float = 0.3
    max_results: int = 20
    layers: list[int] = [1, 2, 3]


@router.post("/api/isomorphisms")
async def find_isomorphisms(request: Request, query: IsomorphismQuery) -> dict[str, Any]:
    driver = request.app.state.driver
    async with driver.session() as session:
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
        results_by_fqn: dict[str, dict[str, Any]] = {}
        topo_hash = target.get("topo_hash")
        if 1 in query.layers and topo_hash:
            topo_result = await session.run(
                """
                MATCH (parent:Atom:Decomposed)
                WHERE parent.topo_hash = $topo_hash AND parent.repo <> $exclude_repo
                MATCH (parent)-[:PARENT_OF]->(child:Atom)
                WITH parent, collect(DISTINCT child.concept_type) AS child_types,
                     count(DISTINCT child) AS n_children
                RETURN parent.fqn AS fqn, parent.repo AS repo,
                       parent.name AS name, parent.concept_type AS concept_type,
                       parent.topo_hash AS topo_hash,
                       n_children, child_types
                ORDER BY n_children DESC
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
        if 2 in query.layers:
            struct_result = await session.run(
                """
                MATCH (candidate:Atom:Decomposed)
                WHERE candidate.repo <> $exclude_repo
                MATCH (candidate)-[:PARENT_OF]->(child:Atom)
                WITH candidate, collect(DISTINCT child.concept_type) AS child_types,
                     count(DISTINCT child) AS n_children
                RETURN candidate.fqn AS fqn, candidate.repo AS repo,
                       candidate.name AS name, candidate.concept_type AS concept_type,
                       candidate.topo_hash AS topo_hash,
                       n_children, child_types
                ORDER BY n_children DESC
                LIMIT $limit
                """,
                parameters={"exclude_repo": target_repo, "limit": query.max_results},
            )
            async for rec in struct_result:
                fqn = rec["fqn"]
                n_in_diff = abs((target.get("n_inputs", 0) or 0) - (rec["n_children"] or 0))
                io_match = 1.0 if n_in_diff == 0 else 0.8
                score = 0.7 * io_match
                existing = results_by_fqn.get(fqn)
                if existing is None or existing["score"] < score:
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
                score = rec["jaccard_score"]
                existing = results_by_fqn.get(fqn)
                if existing:
                    existing["jaccard_score"] = score
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
                        "jaccard_score": score,
                        "layer": 3,
                        "children_summary": rec["child_types"],
                    }
    results = sorted(results_by_fqn.values(), key=lambda row: row["score"], reverse=True)[: query.max_results]
    return {"query_node": query_node, "results": results}
