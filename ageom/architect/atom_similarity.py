"""Pre-decomposition atom similarity: AST fingerprinting and call-graph overlap."""

from __future__ import annotations

import ast
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 1 — Normalised AST fingerprint
# ---------------------------------------------------------------------------


class _AlphaRenamer(ast.NodeTransformer):
    """Rename all local names to canonical ``v0``, ``v1``, … in binding order."""

    def __init__(self) -> None:
        self._map: dict[str, str] = {}
        self._counter = 0

    def _canonical(self, name: str) -> str:
        if name not in self._map:
            self._map[name] = f"v{self._counter}"
            self._counter += 1
        return self._map[name]

    # --- binding sites ---
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self._canonical(node.name)
        # Strip docstring (first Expr(Constant(str)) in body).
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._canonical(node.arg)
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._canonical(node.id)
        return node

    def visit_alias(self, node: ast.alias) -> ast.AST:
        if node.asname:
            node.asname = self._canonical(node.asname)
        return node


def fingerprint_function(source: str) -> str:
    """Return a deterministic SHA-256 hex digest (16 chars) of *source*'s normalised AST.

    The AST is alpha-renamed (locals → ``v0``, ``v1``, …), docstrings are
    stripped, and positional metadata is removed.  Two functions that differ
    only in variable names or comments produce the same fingerprint.
    """
    tree = ast.parse(source)
    tree = ast.fix_missing_locations(_AlphaRenamer().visit(tree))
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Layer 2 — Call-graph extraction & overlap
# ---------------------------------------------------------------------------


class _CallCollector(ast.NodeVisitor):
    """Collect (caller, callee) edges from a function body."""

    def __init__(self) -> None:
        self.edges: list[tuple[str, str]] = []
        self._scope: str = "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self._scope
        self._scope = node.name
        self.generic_visit(node)
        self._scope = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        callee = _resolve_call_name(node)
        if callee:
            self.edges.append((self._scope, callee))
        self.generic_visit(node)


def _resolve_call_name(node: ast.Call) -> str | None:
    """Best-effort name resolution for a Call node."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = [func.attr]
        value = func.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return None


@dataclass
class CallGraph:
    """Lightweight directed graph of (caller → callee) edges."""

    edges: list[tuple[str, str]] = field(default_factory=list)

    @property
    def callees(self) -> set[str]:
        """All unique callee names (leaf names, ignoring dotted prefixes)."""
        result: set[str] = set()
        for _, callee in self.edges:
            result.add(callee)
            # Also add the leaf name for dotted calls like ``np.array``.
            if "." in callee:
                result.add(callee.rsplit(".", 1)[1])
        return result

    @property
    def callers(self) -> set[str]:
        return {caller for caller, _ in self.edges}

    @property
    def nodes(self) -> set[str]:
        return self.callers | self.callees

    def adjacency(self) -> dict[str, set[str]]:
        """Caller → {callees} adjacency list."""
        adj: dict[str, set[str]] = {}
        for caller, callee in self.edges:
            adj.setdefault(caller, set()).add(callee)
        return adj


def extract_call_graph(source: str) -> CallGraph:
    """Parse *source* and return a :class:`CallGraph` of caller→callee edges."""
    tree = ast.parse(source)
    collector = _CallCollector()
    collector.visit(tree)
    return CallGraph(edges=collector.edges)


# ---------------------------------------------------------------------------
# Layer 3 — Combined similarity search
# ---------------------------------------------------------------------------


@dataclass
class SimilarityHit:
    """A single atom match with provenance."""

    atom_name: str
    confidence: float
    match_layer: str  # "fingerprint", "call_overlap", "embedding"
    detail: str = ""


def find_overlapping_atoms(
    source: str,
    catalog_names: set[str],
    *,
    fingerprint_index: dict[str, str] | None = None,
    embedding_fn: object | None = None,
) -> list[SimilarityHit]:
    """Find known atoms that overlap with *source*.

    Parameters
    ----------
    source
        Python source code of the candidate function.
    catalog_names
        Set of known atom names (both short names and fqdns).
    fingerprint_index
        Optional ``{fingerprint_hex: atom_name}`` dict for exact-match lookup.
    embedding_fn
        Reserved for Layer-3 embedding similarity (not yet wired).

    Returns
    -------
    list[SimilarityHit]
        Matches sorted by descending confidence.
    """
    hits: list[SimilarityHit] = []

    # --- Layer 1: exact AST fingerprint ---
    if fingerprint_index:
        try:
            fp = fingerprint_function(source)
        except SyntaxError:
            fp = None
        if fp and fp in fingerprint_index:
            hits.append(
                SimilarityHit(
                    atom_name=fingerprint_index[fp],
                    confidence=1.0,
                    match_layer="fingerprint",
                    detail=f"hash={fp}",
                )
            )

    # --- Layer 2: call-graph overlap ---
    try:
        cg = extract_call_graph(source)
    except SyntaxError:
        return hits

    callees = cg.callees
    if callees and catalog_names:
        # Build a lookup set that includes leaf names from dotted catalog entries.
        catalog_leaves: set[str] = set()
        for name in catalog_names:
            catalog_leaves.add(name)
            if "." in name:
                catalog_leaves.add(name.rsplit(".", 1)[1])

        overlap = callees & catalog_leaves
        if overlap:
            # Map leaf matches back to their full catalog names.
            leaf_to_full: dict[str, str] = {}
            for name in catalog_names:
                leaf_to_full.setdefault(name.rsplit(".", 1)[-1], name)
                leaf_to_full.setdefault(name, name)

            for callee in sorted(overlap):
                full_name = leaf_to_full.get(callee, callee)
                if any(h.atom_name == full_name for h in hits):
                    continue
                hits.append(
                    SimilarityHit(
                        atom_name=full_name,
                        confidence=0.8,
                        match_layer="call_overlap",
                        detail=f"callee={callee}",
                    ),
                )

    hits.sort(key=lambda h: h.confidence, reverse=True)
    return hits


def build_fingerprint_index(
    sources: dict[str, str],
) -> dict[str, str]:
    """Build a ``{fingerprint_hex: atom_name}`` index from atom source code.

    Parameters
    ----------
    sources
        Mapping of ``atom_name`` → Python source text.
    """
    index: dict[str, str] = {}
    for name, src in sources.items():
        try:
            fp = fingerprint_function(src)
            index[fp] = name
        except SyntaxError:
            logger.warning("Cannot fingerprint atom %s: SyntaxError", name)
    return index
