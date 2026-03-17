"""Memgraph graph store for CDG upsert pipeline."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any


def _topo_hash(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], root_id: str) -> str:
    """Compute a topological hash for a decomposed subtree.

    Hash of sorted (in_degree, out_degree) pairs for all children of *root_id*.
    Only counts edges whose *both* endpoints are children of *root_id*,
    excluding phantom edges to/from non-existent nodes (e.g. ``initial``/``final``).
    """
    children = [n for n in nodes if n.get("parent_id") == root_id]
    child_ids = {c["node_id"] for c in children}
    # Only consider edges between sibling children
    sibling_edges = [
        e for e in edges
        if e["source_id"] in child_ids and e["target_id"] in child_ids
    ]
    degree_seq: list[tuple[int, int]] = []
    for cid in sorted(child_ids):
        in_deg = sum(1 for e in sibling_edges if e["target_id"] == cid)
        out_deg = sum(1 for e in sibling_edges if e["source_id"] == cid)
        degree_seq.append((in_deg, out_deg))
    raw = str(sorted(degree_seq))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Witness & contract metadata extraction (pure Python, no Memgraph dependency)
# ---------------------------------------------------------------------------

def extract_witness_metadata(
    repo_path: Path, node_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Parse witness ``.py`` files for annotation types.

    Returns ``{node_id: {witness_name, witness_input_types, witness_output_types,
    abstract_type_class, is_stateful}}`` for each *node_id* that has a witness.
    """
    result: dict[str, dict[str, Any]] = {}
    witness_files = list(repo_path.glob("*witnesses*.py"))
    for wf in witness_files:
        try:
            tree = ast.parse(wf.read_text())
        except SyntaxError:
            continue
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef):
                continue
            if not func.name.startswith("witness_"):
                continue
            # Match witness to node_id: witness_{node_id}
            candidate_id = func.name[len("witness_"):]
            if candidate_id not in node_ids:
                continue

            input_types: list[str] = []
            is_stateful = False
            for arg in func.args.args:
                if arg.annotation:
                    type_name = ast.unparse(arg.annotation)
                    input_types.append(type_name)
                    if "state" in arg.arg.lower():
                        is_stateful = True

            output_types: list[str] = []
            if func.returns:
                ret_str = ast.unparse(func.returns)
                # Flatten tuple[A, B, C] into [A, B, C]
                m = re.match(r"tuple\[(.+)\]", ret_str, re.IGNORECASE)
                if m:
                    output_types = [t.strip() for t in m.group(1).split(",")]
                else:
                    output_types = [ret_str]

            # Dominant abstract type: most common base type across all annotations
            all_types = input_types + output_types
            type_counts: dict[str, int] = {}
            for t in all_types:
                # Strip 'Abstract' prefix for classification
                base = t.replace("Abstract", "")
                type_counts[base] = type_counts.get(base, 0) + 1
            abstract_type_class = max(type_counts, key=type_counts.get) if type_counts else ""

            result[candidate_id] = {
                "witness_name": func.name,
                "witness_input_types": input_types,
                "witness_output_types": output_types,
                "abstract_type_class": abstract_type_class,
                "is_stateful": is_stateful,
            }
    return result


def extract_contract_metadata(
    repo_path: Path, node_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Parse atom ``.py`` files for ``@icontract`` decorators.

    Returns ``{node_id: {input_contracts: [...], output_contracts: [...]}}``
    for each *node_id* that has contracts.
    """
    result: dict[str, dict[str, Any]] = {}
    atom_files = list(repo_path.glob("*atoms*.py"))
    for af in atom_files:
        try:
            tree = ast.parse(af.read_text())
        except SyntaxError:
            continue
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef):
                continue
            if func.name not in node_ids:
                continue

            input_contracts: list[str] = []
            output_contracts: list[str] = []
            for dec in func.decorator_list:
                if not isinstance(dec, ast.Call):
                    continue
                dec_name = ast.unparse(dec.func) if hasattr(dec, "func") else ""
                desc = ""
                # Extract the string argument (description) from decorator
                for kw in dec.keywords:
                    pass
                # icontract passes description as the second positional arg or via lambda
                for arg in dec.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        desc = arg.value
                        break
                if not desc:
                    # Check keywords — no keyword name for positional, but let's check
                    for kw in dec.keywords:
                        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            desc = kw.value.value
                            break

                if "require" in dec_name:
                    input_contracts.append(desc or ast.unparse(dec))
                elif "ensure" in dec_name:
                    output_contracts.append(desc or ast.unparse(dec))

            if input_contracts or output_contracts:
                result[func.name] = {
                    "input_contracts": input_contracts,
                    "output_contracts": output_contracts,
                }
    return result


# ---------------------------------------------------------------------------
# Cypher parameter builders (testable without Memgraph)
# ---------------------------------------------------------------------------

def build_atom_params(
    repo: str,
    node: dict[str, Any],
    witness_meta: dict[str, Any] | None = None,
    contract_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the property dict for an :Atom MERGE."""
    node_id = node["node_id"]
    fqn = f"{repo}.{node_id}"
    inputs = node.get("inputs", []) or []
    outputs = node.get("outputs", []) or []
    props: dict[str, Any] = {
        "node_id": node_id,
        "repo": repo,
        "fqn": fqn,
        "name": node.get("name", node_id),
        "description": node.get("description", ""),
        "concept_type": node.get("concept_type", ""),
        "status": node.get("status", "atomic"),
        "depth": node.get("depth", 0),
        "type_signature": node.get("type_signature", ""),
        "is_optional": bool(node.get("is_optional", False)),
        "is_opaque": bool(node.get("is_opaque", False)),
        "is_external": bool(node.get("is_external", False)),
        "parallelizable": bool(node.get("parallelizable", False)),
        "conceptual_summary": node.get("conceptual_summary", ""),
        "n_inputs": len(inputs),
        "n_outputs": len(outputs),
    }
    # Include verified_leaf_coverage when present
    vlc = node.get("verified_leaf_coverage")
    if vlc is not None:
        props["verified_leaf_coverage"] = float(vlc)
    # Include matched_primitive when present
    mp = node.get("matched_primitive")
    if mp:
        props["matched_primitive"] = str(mp)
    if witness_meta:
        props["witness_name"] = witness_meta.get("witness_name", "")
        props["witness_input_types"] = witness_meta.get("witness_input_types", [])
        props["witness_output_types"] = witness_meta.get("witness_output_types", [])
        props["abstract_type_class"] = witness_meta.get("abstract_type_class", "")
        props["is_stateful"] = witness_meta.get("is_stateful", False)
    if contract_meta:
        props["input_contracts"] = contract_meta.get("input_contracts", [])
        props["output_contracts"] = contract_meta.get("output_contracts", [])
    return props


def build_port_params(
    repo: str, atom_node_id: str, io_spec: dict[str, Any], direction: str
) -> dict[str, Any]:
    """Build property dict for an :InputPort or :OutputPort MERGE."""
    name = io_spec.get("name", "")
    port_id = f"{repo}.{atom_node_id}.{direction}.{name}"
    return {
        "port_id": port_id,
        "name": name,
        "type_desc": io_spec.get("type_desc", ""),
        "constraints": io_spec.get("constraints", ""),
    }


def build_edge_params(edge: dict[str, Any]) -> dict[str, Any]:
    """Build property dict for a :DATA_FLOW relationship."""
    return {
        "output_name": edge.get("output_name", ""),
        "input_name": edge.get("input_name", ""),
        "source_type": edge.get("source_type", ""),
        "target_type": edge.get("target_type", ""),
        "requires_glue": bool(edge.get("requires_glue", False)),
    }


def collect_stale_fqns(
    repo: str,
    cdg_node_ids: set[str],
    all_fqns_in_repo: set[str],
) -> set[str]:
    """Return FQNs in the database that are not in the current CDG."""
    current_fqns = {f"{repo}.{nid}" for nid in cdg_node_ids}
    return all_fqns_in_repo - current_fqns


# ---------------------------------------------------------------------------
# Memgraph async store
# ---------------------------------------------------------------------------

class GraphStore:
    """Async context manager wrapping ``neo4j.AsyncDriver`` (works with Memgraph)."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: Any = None

    async def __aenter__(self) -> "GraphStore":
        from neo4j import AsyncGraphDatabase

        auth = (self._user, self._password) if self._user else None
        self._driver = AsyncGraphDatabase.driver(self._uri, auth=auth)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._driver:
            await self._driver.close()

    async def ensure_constraints(self) -> None:
        """Create uniqueness constraints and indexes (Memgraph DDL)."""
        constraints = [
            "CREATE CONSTRAINT ON (a:Atom) ASSERT a.fqn IS UNIQUE",
            "CREATE CONSTRAINT ON (p:InputPort) ASSERT p.port_id IS UNIQUE",
            "CREATE CONSTRAINT ON (p:OutputPort) ASSERT p.port_id IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX ON :Atom(concept_type)",
            "CREATE INDEX ON :Atom(abstract_type_class)",
            "CREATE INDEX ON :Atom(repo)",
            "CREATE INDEX ON :Atom(topo_hash)",
            "CREATE INDEX ON :Atom(verified_leaf_coverage)",
        ]
        async with self._driver.session() as session:
            for stmt in constraints + indexes:
                try:
                    await session.run(stmt)
                except Exception:
                    pass  # Memgraph has no IF NOT EXISTS; ignore duplicates

    async def query_by_topo_hash(
        self, topo_hash: str, exclude_repo: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Layer 1: exact topo_hash match on Decomposed atoms."""
        cypher = (
            "MATCH (parent:Atom:Decomposed {topo_hash: $topo_hash}) "
            "WHERE parent.repo <> $exclude_repo "
            "MATCH (parent)-[:PARENT_OF]->(child:Atom) "
            "OPTIONAL MATCH (child)-[df:DATA_FLOW]->(sibling:Atom) "
            "  WHERE (parent)-[:PARENT_OF]->(sibling) "
            "WITH parent, "
            "     collect(DISTINCT {node_id: child.node_id, name: child.name, "
            "       description: child.description, concept_type: child.concept_type, "
            "       status: child.status, n_inputs: child.n_inputs, n_outputs: child.n_outputs, "
            "       type_signature: child.type_signature, "
            "       abstract_type_class: child.abstract_type_class, "
            "       matched_primitive: child.matched_primitive, "
            "       witness_input_types: child.witness_input_types, "
            "       witness_output_types: child.witness_output_types}) AS children, "
            "     collect(DISTINCT CASE WHEN df IS NOT NULL THEN {source_id: startNode(df).node_id, "
            "       target_id: endNode(df).node_id, "
            "       output_name: df.output_name, input_name: df.input_name} ELSE NULL END) AS raw_edges "
            "With parent, children, [e IN raw_edges WHERE e IS NOT NULL] AS edges "
            "RETURN parent.fqn AS fqn, parent.name AS name, parent.description AS description, "
            "       parent.concept_type AS concept_type, parent.repo AS repo, "
            "       parent.topo_hash AS topo_hash, "
            "       parent.abstract_type_class AS p_abstract_type_class, "
            "       parent.n_inputs AS p_n_inputs, parent.n_outputs AS p_n_outputs, "
            "       children, edges "
            "LIMIT $limit"
        )
        async with self._driver.session() as session:
            result = await session.run(
                cypher, topo_hash=topo_hash, exclude_repo=exclude_repo, limit=limit
            )
            return [dict(r) async for r in result]

    async def query_by_structure(
        self,
        concept_type: str,
        n_inputs: int,
        n_outputs: int,
        exclude_repo: str,
        min_children: int = 2,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Layer 2: structural match by concept_type and port arity (±1)."""
        cypher = (
            "MATCH (parent:Atom:Decomposed) "
            "WHERE parent.concept_type = $concept_type "
            "  AND parent.repo <> $exclude_repo "
            "  AND abs(parent.n_inputs - $n_inputs) <= 1 "
            "  AND abs(parent.n_outputs - $n_outputs) <= 1 "
            "MATCH (parent)-[:PARENT_OF]->(child:Atom) "
            "WITH parent, collect(child) AS child_list "
            "WHERE size(child_list) >= $min_children "
            "UNWIND child_list AS child "
            "OPTIONAL MATCH (child)-[df:DATA_FLOW]->(sibling:Atom) "
            "  WHERE (parent)-[:PARENT_OF]->(sibling) "
            "WITH parent, "
            "     collect(DISTINCT {node_id: child.node_id, name: child.name, "
            "       description: child.description, concept_type: child.concept_type, "
            "       status: child.status, n_inputs: child.n_inputs, n_outputs: child.n_outputs, "
            "       type_signature: child.type_signature, "
            "       abstract_type_class: child.abstract_type_class, "
            "       matched_primitive: child.matched_primitive, "
            "       witness_input_types: child.witness_input_types, "
            "       witness_output_types: child.witness_output_types}) AS children, "
            "     collect(DISTINCT CASE WHEN df IS NOT NULL THEN {source_id: startNode(df).node_id, "
            "       target_id: endNode(df).node_id, "
            "       output_name: df.output_name, input_name: df.input_name} ELSE NULL END) AS raw_edges "
            "WITH parent, children, [e IN raw_edges WHERE e IS NOT NULL] AS edges "
            "RETURN parent.fqn AS fqn, parent.name AS name, parent.description AS description, "
            "       parent.concept_type AS concept_type, parent.repo AS repo, "
            "       parent.topo_hash AS topo_hash, "
            "       parent.abstract_type_class AS p_abstract_type_class, "
            "       parent.n_inputs AS p_n_inputs, parent.n_outputs AS p_n_outputs, "
            "       children, edges "
            "ORDER BY size(children) DESC "
            "LIMIT $limit"
        )
        async with self._driver.session() as session:
            result = await session.run(
                cypher,
                concept_type=concept_type,
                n_inputs=n_inputs,
                n_outputs=n_outputs,
                exclude_repo=exclude_repo,
                min_children=min_children,
                limit=limit,
            )
            return [dict(r) async for r in result]

    async def query_jaccard_neighborhood(
        self, fqn: str, exclude_repo: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Layer 3: Jaccard pairwise similarity via MAGE node_similarity."""
        cypher = (
            "MATCH (query:Atom {fqn: $fqn})-[:PARENT_OF]->(qc:Atom) "
            "WITH query, collect(qc) AS query_children "
            "WHERE size(query_children) > 0 "
            "MATCH (candidate:Atom:Decomposed)-[:PARENT_OF]->(cc:Atom) "
            "WHERE candidate.repo <> $exclude_repo AND candidate.fqn <> $fqn "
            "WITH query, query_children, candidate, collect(cc) AS cand_children "
            "WHERE size(cand_children) > 0 "
            "WITH query, candidate, "
            "     [qc IN query_children | qc.concept_type] AS q_types, "
            "     [cc IN cand_children | cc.concept_type] AS c_types "
            "WITH candidate, "
            "     toFloat(size([x IN q_types WHERE x IN c_types])) / "
            "     toFloat(size(q_types + [y IN c_types WHERE NOT y IN q_types])) AS jaccard_score "
            "WHERE jaccard_score > 0.3 "
            "WITH candidate, jaccard_score "
            "ORDER BY jaccard_score DESC "
            "LIMIT $limit "
            "MATCH (candidate)-[:PARENT_OF]->(child:Atom) "
            "OPTIONAL MATCH (child)-[df:DATA_FLOW]->(sibling:Atom) "
            "  WHERE (candidate)-[:PARENT_OF]->(sibling) "
            "WITH candidate, jaccard_score, "
            "     collect(DISTINCT {node_id: child.node_id, name: child.name, "
            "       description: child.description, concept_type: child.concept_type, "
            "       status: child.status, n_inputs: child.n_inputs, n_outputs: child.n_outputs, "
            "       type_signature: child.type_signature, "
            "       abstract_type_class: child.abstract_type_class, "
            "       matched_primitive: child.matched_primitive, "
            "       witness_input_types: child.witness_input_types, "
            "       witness_output_types: child.witness_output_types}) AS children, "
            "     collect(DISTINCT CASE WHEN df IS NOT NULL THEN {source_id: startNode(df).node_id, "
            "       target_id: endNode(df).node_id, "
            "       output_name: df.output_name, input_name: df.input_name} ELSE NULL END) AS raw_edges "
            "WITH candidate, jaccard_score, children, [e IN raw_edges WHERE e IS NOT NULL] AS edges "
            "RETURN candidate.fqn AS fqn, candidate.name AS name, candidate.description AS description, "
            "       candidate.concept_type AS concept_type, candidate.repo AS repo, "
            "       candidate.topo_hash AS topo_hash, children, edges, jaccard_score "
        )
        async with self._driver.session() as session:
            result = await session.run(
                cypher, fqn=fqn, exclude_repo=exclude_repo, limit=limit
            )
            return [dict(r) async for r in result]

    async def query_verified_exemplars(
        self,
        concept_type: str,
        min_coverage: float = 0.8,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Query for verified exemplars with high leaf coverage."""
        cypher = (
            "MATCH (p:Atom:Decomposed) "
            "WHERE p.verified_leaf_coverage >= $min_coverage "
            "  AND p.concept_type = $concept_type "
            "RETURN p.fqn AS fqn, p.repo AS repo, "
            "       p.verified_leaf_coverage AS verified_leaf_coverage, "
            "       p.topo_hash AS topo_hash "
            "ORDER BY p.verified_leaf_coverage DESC "
            "LIMIT $limit"
        )
        async with self._driver.session() as session:
            result = await session.run(
                cypher,
                concept_type=concept_type,
                min_coverage=min_coverage,
                limit=limit,
            )
            return [dict(r) async for r in result]

    async def upsert_cdg(
        self,
        repo: str,
        cdg_dict: dict[str, Any],
        witness_meta: dict[str, dict[str, Any]],
        contract_meta: dict[str, dict[str, Any]],
    ) -> dict[str, int]:
        """Idempotent upsert of a single CDG into Memgraph.

        Returns counts: ``{atoms, input_ports, output_ports, data_flow, parent_of, deleted}``.
        """
        nodes = cdg_dict.get("nodes", [])
        edges = cdg_dict.get("edges", [])
        cdg_node_ids = {n["node_id"] for n in nodes}
        counts = {
            "atoms": 0,
            "input_ports": 0,
            "output_ports": 0,
            "data_flow": 0,
            "parent_of": 0,
            "deleted": 0,
            "orphaned_ports": 0,
        }

        async with self._driver.session() as session:
            # 1. MERGE :Atom nodes
            for node in nodes:
                nid = node["node_id"]
                w_meta = witness_meta.get(nid)
                c_meta = contract_meta.get(nid)
                props = build_atom_params(repo, node, w_meta, c_meta)
                status = node.get("status", "atomic")
                concept_type = node.get("concept_type", "")

                # Build topo_hash for decomposed nodes
                topo_hash = ""
                if status == "decomposed":
                    topo_hash = _topo_hash(nodes, edges, nid)

                # MERGE atom, SET all properties, add secondary labels
                cypher = (
                    "MERGE (a:Atom {fqn: $fqn}) "
                    "SET a += $props "
                )
                if topo_hash:
                    cypher += "SET a.topo_hash = $topo_hash "

                # Add status label
                if status == "atomic":
                    cypher += "SET a:Atomic REMOVE a:Decomposed "
                else:
                    cypher += "SET a:Decomposed REMOVE a:Atomic "

                # Add concept_type label
                if concept_type:
                    # Graph labels can't have hyphens; replace with underscore
                    safe_label = concept_type.replace("-", "_").replace(" ", "_")
                    cypher += f"SET a:`{safe_label}` "

                await session.run(
                    cypher,
                    fqn=props["fqn"],
                    props=props,
                    topo_hash=topo_hash,
                )
                counts["atoms"] += 1

            # 2. MERGE :InputPort / :OutputPort + HAS_INPUT / HAS_OUTPUT
            expected_port_ids: set[str] = set()
            for node in nodes:
                nid = node["node_id"]
                fqn = f"{repo}.{nid}"
                for io_spec in node.get("inputs", []) or []:
                    port_props = build_port_params(repo, nid, io_spec, "in")
                    expected_port_ids.add(port_props["port_id"])
                    await session.run(
                        "MERGE (p:InputPort {port_id: $port_id}) "
                        "SET p += $props "
                        "WITH p "
                        "MATCH (a:Atom {fqn: $fqn}) "
                        "MERGE (a)-[:HAS_INPUT]->(p)",
                        port_id=port_props["port_id"],
                        props=port_props,
                        fqn=fqn,
                    )
                    counts["input_ports"] += 1

                for io_spec in node.get("outputs", []) or []:
                    port_props = build_port_params(repo, nid, io_spec, "out")
                    expected_port_ids.add(port_props["port_id"])
                    await session.run(
                        "MERGE (p:OutputPort {port_id: $port_id}) "
                        "SET p += $props "
                        "WITH p "
                        "MATCH (a:Atom {fqn: $fqn}) "
                        "MERGE (a)-[:HAS_OUTPUT]->(p)",
                        port_id=port_props["port_id"],
                        props=port_props,
                        fqn=fqn,
                    )
                    counts["output_ports"] += 1

            # 2b. DELETE orphaned ports (stale from previous I/O spec)
            if expected_port_ids:
                result = await session.run(
                    "MATCH (a:Atom {repo: $repo})-[r:HAS_INPUT|HAS_OUTPUT]->(p) "
                    "WHERE NOT p.port_id IN $expected_port_ids "
                    "DETACH DELETE p "
                    "RETURN count(p) AS cnt",
                    repo=repo,
                    expected_port_ids=list(expected_port_ids),
                )
                record = await result.single()
                counts["orphaned_ports"] = record["cnt"] if record else 0
            else:
                counts["orphaned_ports"] = 0

            # 3. MERGE PARENT_OF edges
            for node in nodes:
                if not node.get("children"):
                    continue
                parent_fqn = f"{repo}.{node['node_id']}"
                for child_id in node["children"]:
                    if child_id not in cdg_node_ids:
                        continue
                    child_fqn = f"{repo}.{child_id}"
                    await session.run(
                        "MATCH (p:Atom {fqn: $parent_fqn}) "
                        "MATCH (c:Atom {fqn: $child_fqn}) "
                        "MERGE (p)-[:PARENT_OF]->(c)",
                        parent_fqn=parent_fqn,
                        child_fqn=child_fqn,
                    )
                    counts["parent_of"] += 1

            # 4. MERGE DATA_FLOW edges
            for edge in edges:
                source_fqn = f"{repo}.{edge['source_id']}"
                target_fqn = f"{repo}.{edge['target_id']}"
                edge_props = build_edge_params(edge)
                await session.run(
                    "MATCH (s:Atom {fqn: $source_fqn}) "
                    "MATCH (t:Atom {fqn: $target_fqn}) "
                    "MERGE (s)-[r:DATA_FLOW {output_name: $output_name, input_name: $input_name}]->(t) "
                    "SET r += $props",
                    source_fqn=source_fqn,
                    target_fqn=target_fqn,
                    output_name=edge_props["output_name"],
                    input_name=edge_props["input_name"],
                    props=edge_props,
                )
                counts["data_flow"] += 1

            # 5. DELETE stale nodes from previous CDG version (same repo scope)
            # Find all Atom FQNs in this repo that are NOT in the current CDG
            result = await session.run(
                "MATCH (a:Atom {repo: $repo}) RETURN a.fqn AS fqn",
                repo=repo,
            )
            records = [r async for r in result]
            all_fqns = {r["fqn"] for r in records}
            stale = collect_stale_fqns(repo, cdg_node_ids, all_fqns)
            if stale:
                # Delete stale ports first, then atoms
                await session.run(
                    "MATCH (a:Atom) WHERE a.fqn IN $stale "
                    "OPTIONAL MATCH (a)-[:HAS_INPUT]->(ip:InputPort) "
                    "OPTIONAL MATCH (a)-[:HAS_OUTPUT]->(op:OutputPort) "
                    "DETACH DELETE ip, op, a",
                    stale=list(stale),
                )
                counts["deleted"] = len(stale)

        return counts
