"""Ingest coq-100-theorems into the primitive catalog.

Parses the rocq-community/coq-100-theorems repo to extract theorem metadata,
converting each formalized theorem into an AlgorithmicPrimitive.
"""

from __future__ import annotations

from pathlib import Path

from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.models import AlgorithmicPrimitive, ConceptType, IOSpec

# Keyword heuristics for mapping theorem topics to ConceptType
_KEYWORD_RULES: list[tuple[list[str], ConceptType]] = [
    (
        ["prime", "divisib", "euclid", "fermat", "gcd", "lcm", "modular", "diophant"],
        ConceptType.NUMBER_THEORY,
    ),
    (
        [
            "sqrt",
            "irrational",
            "real",
            "limit",
            "continu",
            "deriv",
            "integral",
            "mean_value",
            "taylor",
            "cauchy",
            "convergent",
            "monotone",
            "bounded",
        ],
        ConceptType.ANALYSIS,
    ),
    (
        [
            "algebra",
            "polynomial",
            "ring",
            "field",
            "group",
            "linear",
            "eigenvalue",
            "matrix",
            "vector",
            "fundament.*algebra",
            "hermite",
        ],
        ConceptType.ALGEBRA,
    ),
    (
        [
            "triangle",
            "circle",
            "angle",
            "area",
            "polygon",
            "pythagor",
            "desargues",
            "feuerbach",
            "heron",
            "isosceles",
        ],
        ConceptType.GEOMETRY,
    ),
    (
        [
            "combinat",
            "binomial",
            "pigeon",
            "ramsey",
            "partition",
            "permutation",
            "ballot",
            "derangement",
            "catalan",
        ],
        ConceptType.COMBINATORICS,
    ),
    (
        ["set", "cardinal", "cantor", "countab", "power_set", "schroeder", "zorn"],
        ConceptType.SET_THEORY,
    ),
    (
        ["sum", "product", "arith", "number", "infinity_of_primes", "perfect"],
        ConceptType.ARITHMETIC,
    ),
    (["sort", "order"], ConceptType.SORTING),
    (["graph", "euler", "konigsberg"], ConceptType.GRAPH_TRAVERSAL),
]


def _classify_theorem(name: str, statement: str = "") -> ConceptType:
    """Classify a theorem into a ConceptType using keyword heuristics."""
    text = f"{name} {statement}".lower()
    for keywords, concept_type in _KEYWORD_RULES:
        for kw in keywords:
            if kw in text:
                return concept_type
    return ConceptType.CUSTOM


def _parse_yaml(path: Path) -> dict | list:
    """Parse a YAML file. Uses PyYAML if available, falls back to basic parsing."""
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return _basic_yaml_parse(path)


def _basic_yaml_parse(path: Path) -> dict | list:
    """Minimal YAML parser for simple key-value or list structures."""
    text = path.read_text()
    result: dict = {}
    current_key: str | None = None
    current_value_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Top-level key: "123:" or "theorem_name:"
        if not line.startswith((" ", "\t")) and ":" in stripped:
            if current_key is not None:
                result[current_key] = "\n".join(current_value_lines).strip()
            key, _, value = stripped.partition(":")
            current_key = key.strip()
            current_value_lines = [value.strip()] if value.strip() else []
        elif current_key is not None:
            # Continuation of a multi-line value
            current_value_lines.append(stripped)

    if current_key is not None:
        result[current_key] = "\n".join(current_value_lines).strip()

    return result


def ingest_coq100(coq100_path: str | Path) -> PrimitiveCatalog:
    """Ingest coq-100-theorems into a PrimitiveCatalog.

    Args:
        coq100_path: Path to the cloned rocq-community/coq-100-theorems repo.

    Returns:
        A PrimitiveCatalog containing AlgorithmicPrimitives for each theorem.
    """
    coq100_path = Path(coq100_path)
    catalog = PrimitiveCatalog()

    # Primary: statements.yml (formal Coq statements)
    statements_path = coq100_path / "statements.yml"
    theorems_path = coq100_path / "theorems.yml"

    statements: dict = {}
    if statements_path.exists():
        raw = _parse_yaml(statements_path)
        if isinstance(raw, dict):
            statements = raw

    # Secondary: theorems.yml (numbered theorem names)
    theorem_names: dict = {}
    if theorems_path.exists():
        raw = _parse_yaml(theorems_path)
        if isinstance(raw, dict):
            theorem_names = raw

    # Merge: use statements as primary, fill names from theorems.yml
    all_keys = set(list(statements.keys()) + list(theorem_names.keys()))

    for key in sorted(all_keys, key=lambda k: str(k)):
        name = str(theorem_names.get(key, key))
        statement = str(statements.get(key, ""))

        # Clean up the name
        if name.startswith('"') and name.endswith('"'):
            name = name[1:-1]
        if statement.startswith('"') and statement.endswith('"'):
            statement = statement[1:-1]

        concept_type = _classify_theorem(name, statement)

        # Build IOSpec from the statement if it looks like a Coq type
        outputs: list[IOSpec] = []
        if statement:
            outputs = [IOSpec(name="statement", type_desc=statement[:200])]

        primitive = AlgorithmicPrimitive(
            name=f"thm_{key}" if str(key).isdigit() else str(key),
            source="coq-100-theorems",
            category=concept_type,
            description=f"Theorem: {name}",
            inputs=[],
            outputs=outputs,
            type_signature=statement,
        )
        catalog.add(primitive)

    return catalog
