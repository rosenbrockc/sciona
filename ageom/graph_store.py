"""Neo4j graph store for CDG upsert pipeline."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any


def _topo_hash(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], root_id: str) -> str:
    """Compute a topological hash for a decomposed subtree.

    Hash of sorted (in_degree, out_degree) pairs for all children of *root_id*.
    """
    children = [n for n in nodes if n.get("parent_id") == root_id]
    child_ids = {c["node_id"] for c in children}
    degree_seq: list[tuple[int, int]] = []
    for cid in sorted(child_ids):
        in_deg = sum(1 for e in edges if e["target_id"] == cid)
        out_deg = sum(1 for e in edges if e["source_id"] == cid)
        degree_seq.append((in_deg, out_deg))
    raw = str(sorted(degree_seq))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Witness & contract metadata extraction (pure Python, no Neo4j dependency)
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
# Cypher parameter builders (testable without Neo4j)
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
# Neo4j async store
# ---------------------------------------------------------------------------

class Neo4jStore:
    """Async context manager wrapping ``neo4j.AsyncDriver``."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: Any = None

    async def __aenter__(self) -> "Neo4jStore":
        from neo4j import AsyncGraphDatabase

        self._driver = AsyncGraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._driver:
            await self._driver.close()

    async def ensure_constraints(self) -> None:
        """Create uniqueness constraints and indexes."""
        statements = [
            "CREATE CONSTRAINT atom_fqn IF NOT EXISTS FOR (a:Atom) REQUIRE a.fqn IS UNIQUE",
            "CREATE CONSTRAINT input_port_id IF NOT EXISTS FOR (p:InputPort) REQUIRE p.port_id IS UNIQUE",
            "CREATE CONSTRAINT output_port_id IF NOT EXISTS FOR (p:OutputPort) REQUIRE p.port_id IS UNIQUE",
            "CREATE INDEX atom_concept_type IF NOT EXISTS FOR (a:Atom) ON (a.concept_type)",
            "CREATE INDEX atom_abstract_type IF NOT EXISTS FOR (a:Atom) ON (a.abstract_type_class)",
            "CREATE INDEX atom_repo IF NOT EXISTS FOR (a:Atom) ON (a.repo)",
        ]
        async with self._driver.session() as session:
            for stmt in statements:
                await session.run(stmt)

    async def upsert_cdg(
        self,
        repo: str,
        cdg_dict: dict[str, Any],
        witness_meta: dict[str, dict[str, Any]],
        contract_meta: dict[str, dict[str, Any]],
    ) -> dict[str, int]:
        """Idempotent upsert of a single CDG into Neo4j.

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
                    # Neo4j labels can't have hyphens; replace with underscore
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
            for node in nodes:
                nid = node["node_id"]
                fqn = f"{repo}.{nid}"
                for io_spec in node.get("inputs", []) or []:
                    port_props = build_port_params(repo, nid, io_spec, "in")
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
