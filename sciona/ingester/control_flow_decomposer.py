"""Deterministic control-flow-based function decomposition.

Splits complex functions into sub-atoms at structural boundaries (function
calls, loops, conditional branches) using Python AST analysis. Provides a
deterministic fallback for INGESTER_DECOMPOSE when the function structure
is straightforward enough to split without LLM assistance.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubAtom:
    """A sub-step extracted from a function's control flow."""

    name: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    source_lines: tuple[int, int] = (0, 0)


@dataclass
class DecomposeResult:
    """Result of control-flow decomposition."""

    sub_atoms: list[SubAtom]
    edges: list[dict[str, str]]
    confidence: float  # 0.0-1.0, how confident we are this is a good split


_STOPWORDS = {"self", "cls", "args", "kwargs", "None", "True", "False"}


def _reads_writes(node: ast.AST) -> tuple[set[str], set[str]]:
    """Extract variable reads and writes from an AST node."""
    reads: set[str] = set()
    writes: set[str] = set()

    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                for name_node in ast.walk(target):
                    if isinstance(name_node, ast.Name):
                        writes.add(name_node.id)
        elif isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name):
            writes.add(child.target.id)
            reads.add(child.target.id)
        elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            reads.add(child.id)

    reads -= _STOPWORDS
    writes -= _STOPWORDS
    return reads, writes


def _call_name(node: ast.Call) -> str:
    """Extract the function name from a Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _snake_to_title(name: str) -> str:
    return " ".join(word.capitalize() for word in name.replace("_", " ").split())


def _describe_block(stmts: list[ast.stmt]) -> str:
    """Generate a brief description of a block of statements."""
    parts = []
    for stmt in stmts:
        if isinstance(stmt, ast.Assign) or isinstance(stmt, ast.AugAssign):
            parts.append("compute")
        elif isinstance(stmt, ast.For):
            parts.append("iterate")
        elif isinstance(stmt, ast.While):
            parts.append("loop")
        elif isinstance(stmt, ast.If):
            parts.append("branch")
        elif isinstance(stmt, ast.Return):
            parts.append("return result")
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            name = _call_name(stmt.value)
            if name:
                parts.append(f"call {name}")
    return "; ".join(parts[:4]) if parts else "process data"


class _FunctionSplitter(ast.NodeVisitor):
    """Splits a function body into logical sub-atoms at structural boundaries."""

    def __init__(self, min_block_size: int = 3) -> None:
        self._min_block_size = min_block_size
        self.blocks: list[tuple[str, list[ast.stmt], set[str], set[str]]] = []

    def split(self, func_body: list[ast.stmt]) -> list[tuple[str, list[ast.stmt], set[str], set[str]]]:
        """Split function body into named blocks with their reads/writes."""
        current_block: list[ast.stmt] = []
        block_idx = 0

        for stmt in func_body:
            # Split at major structural boundaries
            if isinstance(stmt, (ast.For, ast.While)):
                if current_block:
                    self._emit_block(f"setup_{block_idx}", current_block)
                    block_idx += 1
                    current_block = []
                loop_name = "iterate" if isinstance(stmt, ast.For) else "loop"
                # Check if the loop has a recognizable target
                if isinstance(stmt, ast.For) and isinstance(stmt.iter, ast.Call):
                    call_name = _call_name(stmt.iter)
                    if call_name:
                        loop_name = f"{loop_name}_{call_name}"
                self._emit_block(f"{loop_name}_{block_idx}", [stmt])
                block_idx += 1

            elif isinstance(stmt, ast.If) and len(stmt.body) >= self._min_block_size:
                if current_block:
                    self._emit_block(f"prepare_{block_idx}", current_block)
                    block_idx += 1
                    current_block = []
                self._emit_block(f"branch_{block_idx}", [stmt])
                block_idx += 1

            elif (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                  and _call_name(stmt.value)):
                # Significant function call as a standalone statement
                call_name = _call_name(stmt.value)
                if current_block and len(current_block) >= self._min_block_size:
                    self._emit_block(f"prepare_{block_idx}", current_block)
                    block_idx += 1
                    current_block = []
                    self._emit_block(f"call_{call_name}_{block_idx}", [stmt])
                    block_idx += 1
                else:
                    current_block.append(stmt)

            elif isinstance(stmt, ast.Assign):
                # Check if this is a significant call assignment
                if isinstance(stmt.value, ast.Call) and _call_name(stmt.value):
                    call_name = _call_name(stmt.value)
                    if current_block and len(current_block) >= self._min_block_size:
                        self._emit_block(f"prepare_{block_idx}", current_block)
                        block_idx += 1
                        current_block = []
                        self._emit_block(f"compute_{call_name}_{block_idx}", [stmt])
                        block_idx += 1
                    else:
                        current_block.append(stmt)
                else:
                    current_block.append(stmt)

            else:
                current_block.append(stmt)

        if current_block:
            self._emit_block(f"finalize_{block_idx}", current_block)

        return self.blocks

    def _emit_block(self, name: str, stmts: list[ast.stmt]) -> None:
        reads, writes = set(), set()
        for stmt in stmts:
            r, w = _reads_writes(stmt)
            reads |= r
            writes |= w
        self.blocks.append((name, stmts, reads, writes))


def decompose_function(
    source_code: str,
    function_name: str,
    *,
    min_block_size: int = 3,
    min_sub_atoms: int = 2,
    max_sub_atoms: int = 6,
) -> DecomposeResult | None:
    """Attempt to decompose a function into sub-atoms using control-flow analysis.

    Returns None if the function is too simple or can't be parsed.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None

    # Find the target function
    target_func: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                target_func = node
                break

    if target_func is None:
        return None

    body = target_func.body
    # Skip docstring
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, (ast.Constant, ast.Str)):
        body = body[1:]

    if len(body) < min_block_size * min_sub_atoms:
        return None

    splitter = _FunctionSplitter(min_block_size=min_block_size)
    blocks = splitter.split(body)

    if len(blocks) < min_sub_atoms:
        return None

    # Merge tiny adjacent blocks
    merged: list[tuple[str, list[ast.stmt], set[str], set[str]]] = []
    for name, stmts, reads, writes in blocks:
        if merged and len(stmts) < min_block_size and len(merged[-1][1]) < min_block_size:
            prev_name, prev_stmts, prev_reads, prev_writes = merged[-1]
            merged[-1] = (prev_name, prev_stmts + stmts, prev_reads | reads, prev_writes | writes)
        else:
            merged.append((name, stmts, reads, writes))

    if len(merged) < min_sub_atoms:
        return None

    # Cap at max_sub_atoms by merging the smallest blocks
    while len(merged) > max_sub_atoms:
        # Find smallest adjacent pair to merge
        min_size = float("inf")
        min_idx = 0
        for i in range(len(merged) - 1):
            combined = len(merged[i][1]) + len(merged[i + 1][1])
            if combined < min_size:
                min_size = combined
                min_idx = i
        a = merged[min_idx]
        b = merged[min_idx + 1]
        merged[min_idx] = (a[0], a[1] + b[1], a[2] | b[2], a[3] | b[3])
        del merged[min_idx + 1]

    # Build sub-atoms
    sub_atoms: list[SubAtom] = []
    for name, stmts, reads, writes in merged:
        first_line = stmts[0].lineno if stmts else 0
        last_line = stmts[-1].end_lineno if stmts and hasattr(stmts[-1], "end_lineno") else first_line
        sub_atoms.append(SubAtom(
            name=_snake_to_title(name),
            description=_describe_block(stmts),
            inputs=sorted(reads - _STOPWORDS),
            outputs=sorted(writes - _STOPWORDS),
            source_lines=(first_line, last_line or first_line),
        ))

    # Build data-flow edges between consecutive sub-atoms
    edges: list[dict[str, str]] = []
    for i in range(len(sub_atoms) - 1):
        shared = set(sub_atoms[i].outputs) & set(sub_atoms[i + 1].inputs)
        if shared:
            edges.append({
                "from": sub_atoms[i].name,
                "to": sub_atoms[i + 1].name,
                "data": ", ".join(sorted(shared)),
            })
        else:
            # Still add sequential edge even without explicit data flow
            edges.append({
                "from": sub_atoms[i].name,
                "to": sub_atoms[i + 1].name,
                "data": "sequence",
            })

    confidence = min(1.0, len(sub_atoms) / max_sub_atoms + 0.3)
    return DecomposeResult(sub_atoms=sub_atoms, edges=edges, confidence=confidence)


def to_json(result: DecomposeResult) -> dict[str, Any]:
    """Convert a DecomposeResult to JSON-compatible dict matching ingester format."""
    return {
        "sub_atoms": [
            {
                "name": atom.name,
                "description": atom.description,
                "inputs": [{"name": i, "type": ""} for i in atom.inputs],
                "outputs": [{"name": o, "type": ""} for o in atom.outputs],
            }
            for atom in result.sub_atoms
        ],
        "edges": result.edges,
    }
