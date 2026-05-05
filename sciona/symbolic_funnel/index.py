"""In-memory index of symbolic atom entries for fast funnel lookups."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FunnelAtomEntry:
    """One symbolic expression indexed for funnel matching."""

    expression_id: str
    atom_name: str
    atom_module: str
    srepr_str: str
    variables: dict[str, str]  # symbol -> role (input/output/parameter/constant)
    constants: dict[str, float]
    dim_signature: dict[str, str]  # symbol -> compact dim string
    validity_bounds: dict[str, tuple[float | None, float | None]]
    equivalence_class_hash: str
    is_equivalence_representative: bool
    exponent_signature: dict[str, str] | None
    exponent_signature_hash: str | None
    invariant_forms: list[dict[str, Any]] | None
    mechanism_tags: list[str]
    topology_hash: str
    dimensional_hash: str

    @property
    def input_variables(self) -> dict[str, str]:
        """Variables with role 'input' — the data-backed columns."""
        return {k: v for k, v in self.variables.items() if v == "input"}

    @property
    def output_variables(self) -> dict[str, str]:
        """Variables with role 'output'."""
        return {k: v for k, v in self.variables.items() if v == "output"}

    @property
    def constant_variables(self) -> dict[str, str]:
        """Variables with role 'constant'."""
        return {k: v for k, v in self.variables.items() if v == "constant"}

    @property
    def n_unknowns(self) -> int:
        """Number of unknown constants (constants without known values)."""
        return sum(
            1
            for name, role in self.variables.items()
            if role == "constant" and name not in self.constants
        )


@dataclass
class FunnelIndex:
    """In-memory index loaded from publication manifest JSON files.

    Provides O(1) lookup by exponent signature hash and fast iteration
    over equivalence class representatives.
    """

    entries: list[FunnelAtomEntry] = field(default_factory=list)
    by_exponent_hash: dict[str, list[FunnelAtomEntry]] = field(default_factory=dict)
    by_equivalence_class: dict[str, list[FunnelAtomEntry]] = field(default_factory=dict)
    representatives: list[FunnelAtomEntry] = field(default_factory=list)

    @classmethod
    def from_publication_manifests(cls, manifest_dir: Path) -> FunnelIndex:
        """Load all ``*.publication_manifest.json`` files from *manifest_dir*."""
        index = cls()
        for path in sorted(manifest_dir.glob("*.publication_manifest.json")):
            with open(path) as f:
                manifest = json.load(f)
            expressions = manifest.get("artifact_symbolic_expressions", [])
            bounds_by_expr = _group_bounds(
                manifest.get("artifact_validity_bounds", [])
            )
            for row in expressions:
                entry = _row_to_entry(row, bounds_by_expr)
                index._add(entry)
        return index

    @classmethod
    def from_expression_rows(
        cls,
        expression_rows: list[dict[str, Any]],
        bounds_rows: list[dict[str, Any]] | None = None,
    ) -> FunnelIndex:
        """Build index directly from manifest expression rows."""
        index = cls()
        bounds_by_expr = _group_bounds(bounds_rows or [])
        for row in expression_rows:
            entry = _row_to_entry(row, bounds_by_expr)
            index._add(entry)
        return index

    def _add(self, entry: FunnelAtomEntry) -> None:
        self.entries.append(entry)

        eq_hash = entry.equivalence_class_hash
        self.by_equivalence_class.setdefault(eq_hash, []).append(entry)
        if entry.is_equivalence_representative:
            self.representatives.append(entry)

        if entry.exponent_signature_hash is not None:
            self.by_exponent_hash.setdefault(
                entry.exponent_signature_hash, []
            ).append(entry)


def _group_bounds(
    bounds_rows: list[dict[str, Any]],
) -> dict[str, dict[str, tuple[float | None, float | None]]]:
    """Group validity bounds by expression_id."""
    result: dict[str, dict[str, tuple[float | None, float | None]]] = {}
    for row in bounds_rows:
        expr_id = row.get("expression_id", "")
        symbol = row.get("symbol", row.get("variable_name", ""))
        lower = row.get("min_value", row.get("lower_value"))
        upper = row.get("max_value", row.get("upper_value"))
        result.setdefault(expr_id, {})[symbol] = (lower, upper)
    return result


def _row_to_entry(
    row: dict[str, Any],
    bounds_by_expr: dict[str, dict[str, tuple[float | None, float | None]]],
) -> FunnelAtomEntry:
    """Convert a publication manifest expression row to a FunnelAtomEntry."""
    expr_id = row.get("expression_id", "")
    return FunnelAtomEntry(
        expression_id=expr_id,
        atom_name=row.get("atom_name", ""),
        atom_module=row.get("atom_module", ""),
        srepr_str=row.get("expression_srepr", row.get("sympy_srepr", "")),
        variables=row.get("variables", {}),
        constants=row.get("constants", {}),
        dim_signature=row.get("dim_signature", {}),
        validity_bounds=bounds_by_expr.get(expr_id, {}),
        equivalence_class_hash=row.get("equivalence_class_hash", ""),
        is_equivalence_representative=row.get(
            "equivalence_class_representative", True
        ),
        exponent_signature=row.get("exponent_signature"),
        exponent_signature_hash=row.get("exponent_signature_hash"),
        invariant_forms=row.get("invariant_forms"),
        mechanism_tags=row.get("mechanism_tags", []),
        topology_hash=row.get("topology_hash", ""),
        dimensional_hash=row.get("dimensional_hash", ""),
    )
