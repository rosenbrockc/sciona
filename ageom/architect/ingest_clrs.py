"""Ingest CLRS-30 algorithms into the primitive catalog.

Parses the google-deepmind/clrs repo to extract algorithm specs and metadata,
converting each of the 30 classical algorithms into an AlgorithmicPrimitive.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.models import AlgorithmicPrimitive, ConceptType, IOSpec

# Map CLRS source module filenames to ConceptType
_MODULE_TO_CONCEPT: dict[str, ConceptType] = {
    "sorting": ConceptType.SORTING,
    "searching": ConceptType.SEARCHING,
    "divide_and_conquer": ConceptType.DIVIDE_AND_CONQUER,
    "greedy": ConceptType.GREEDY,
    "dynamic_programming": ConceptType.DYNAMIC_PROGRAMMING,
    "graphs": ConceptType.GRAPH_TRAVERSAL,
    "strings": ConceptType.STRING_MATCHING,
    "geometry": ConceptType.GEOMETRY,
}

# CLRS Stage/Location/Type constants (mirrors clrs._src.specs)
_STAGE_INPUT = "input"
_STAGE_OUTPUT = "output"
_STAGE_HINT = "hint"

# CLRS Type names to readable type descriptions
_CLRS_TYPE_MAP: dict[str, str] = {
    "SCALAR": "float",
    "MASK": "bool",
    "MASK_ONE": "one-hot bool",
    "CATEGORICAL": "int (categorical)",
    "POINTER": "int (pointer)",
    "PERMUTATION_POINTER": "int (permutation pointer)",
    "SHOULD_BE_PERMUTATION": "int (permutation)",
    "SOFT_POINTER": "float (soft pointer)",
}


def _parse_specs_dict(specs_path: Path) -> dict:
    """Parse the SPECS dictionary from clrs/_src/specs.py using AST.

    Returns the raw dict structure: {algo_name: {field: (stage, location, type)}}.
    We parse the AST rather than importing to avoid requiring clrs as a dependency.
    """
    source = specs_path.read_text()
    tree = ast.parse(source)

    # Find the SPECS assignment (it's a module-level dict)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SPECS":
                    # Evaluate the dict literal safely
                    # The SPECS dict references module constants, so we need
                    # to resolve them. Instead, we'll use a regex approach.
                    break

    # Fallback: try importing the module directly
    return _import_specs(specs_path)


def _import_specs(specs_path: Path) -> dict:
    """Import clrs specs module to get the SPECS dict."""
    # Add parent dirs to path so clrs imports work
    clrs_root = specs_path.parent.parent.parent
    src_dir = specs_path.parent

    old_path = sys.path[:]
    sys.path.insert(0, str(clrs_root))
    sys.path.insert(0, str(src_dir))

    try:
        spec = importlib.util.spec_from_file_location("specs", specs_path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "SPECS", {})
    except Exception:
        return {}
    finally:
        sys.path[:] = old_path


def _parse_algorithm_files(algorithms_dir: Path) -> dict[str, dict]:
    """Parse algorithm source files for docstrings and function names.

    Returns: {algo_name: {"module": module_name, "docstring": str, "functions": [str]}}
    """
    result: dict[str, dict] = {}

    for py_file in sorted(algorithms_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = py_file.stem
        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            continue

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node) or ""
                result[node.name] = {
                    "module": module_name,
                    "docstring": docstring,
                    "functions": [node.name],
                }

    return result


def _spec_to_io(
    spec_dict: dict, stage: str
) -> list[IOSpec]:
    """Extract IOSpec list from CLRS spec entries of a given stage."""
    ios: list[IOSpec] = []
    for field_name, spec_tuple in spec_dict.items():
        # spec_tuple is (Stage, Location, Type) — but the actual format
        # varies between raw dict and parsed module. Handle both.
        if isinstance(spec_tuple, (list, tuple)) and len(spec_tuple) >= 3:
            s, _loc, t = spec_tuple[0], spec_tuple[1], spec_tuple[2]
            # Stage values: 0=input, 1=output, 2=hint (in the enum)
            stage_name = str(s).lower()
            if stage not in stage_name and str(s) != stage:
                continue
            type_name = str(t).split(".")[-1] if "." in str(t) else str(t)
            type_desc = _CLRS_TYPE_MAP.get(type_name, type_name)
            ios.append(IOSpec(name=field_name, type_desc=type_desc))
        elif hasattr(spec_tuple, "name"):
            # It's an actual enum from the clrs module
            stage_val = spec_tuple[0]
            type_val = spec_tuple[2]
            stage_name = stage_val.name.lower() if hasattr(stage_val, "name") else str(stage_val)
            if stage not in stage_name:
                continue
            type_name = type_val.name if hasattr(type_val, "name") else str(type_val)
            type_desc = _CLRS_TYPE_MAP.get(type_name, type_name)
            ios.append(IOSpec(name=field_name, type_desc=type_desc))

    return ios


def ingest_clrs(clrs_path: str | Path) -> PrimitiveCatalog:
    """Ingest CLRS-30 algorithms into a PrimitiveCatalog.

    Args:
        clrs_path: Path to the cloned google-deepmind/clrs repo.

    Returns:
        A PrimitiveCatalog containing up to 30 AlgorithmicPrimitives.
    """
    clrs_path = Path(clrs_path)
    specs_path = clrs_path / "clrs" / "_src" / "specs.py"
    algorithms_dir = clrs_path / "clrs" / "_src" / "algorithms"

    catalog = PrimitiveCatalog()

    # Parse SPECS dict
    specs: dict = {}
    if specs_path.exists():
        specs = _parse_specs_dict(specs_path)

    # Parse algorithm source files
    algo_meta: dict[str, dict] = {}
    if algorithms_dir.exists():
        algo_meta = _parse_algorithm_files(algorithms_dir)

    # Build primitives from specs (primary source)
    for algo_name, algo_spec in specs.items():
        module_name = algo_meta.get(algo_name, {}).get("module", "")
        docstring = algo_meta.get(algo_name, {}).get("docstring", "")
        concept_type = _MODULE_TO_CONCEPT.get(module_name, ConceptType.CUSTOM)

        # If we don't know the module from function defs, try to infer from algo name
        if concept_type == ConceptType.CUSTOM:
            concept_type = _infer_concept_type(algo_name)

        inputs = _spec_to_io(algo_spec, "input")
        outputs = _spec_to_io(algo_spec, "output")

        description = docstring or f"CLRS-30 algorithm: {algo_name}"

        primitive = AlgorithmicPrimitive(
            name=algo_name,
            source="clrs-30",
            category=concept_type,
            description=description,
            inputs=inputs,
            outputs=outputs,
            clrs_spec={k: str(v) for k, v in algo_spec.items()},
        )
        catalog.add(primitive)

    # Add any algorithms found in source but not in specs
    for func_name, meta in algo_meta.items():
        if func_name not in specs and not func_name.startswith("_"):
            module_name = meta.get("module", "")
            concept_type = _MODULE_TO_CONCEPT.get(module_name, ConceptType.CUSTOM)
            if concept_type == ConceptType.CUSTOM:
                concept_type = _infer_concept_type(func_name)

            primitive = AlgorithmicPrimitive(
                name=func_name,
                source="clrs-30",
                category=concept_type,
                description=meta.get("docstring", f"CLRS algorithm: {func_name}"),
                inputs=[],
                outputs=[],
            )
            catalog.add(primitive)

    return catalog


def _infer_concept_type(name: str) -> ConceptType:
    """Infer concept type from algorithm name using keyword heuristics."""
    name_lower = name.lower()

    sort_keywords = {"sort", "heap", "insertion", "bubble", "merge", "quick"}
    search_keywords = {"search", "find", "binary_search", "minimum"}
    graph_keywords = {
        "bfs", "dfs", "dijkstra", "bellman", "floyd", "warshall", "prim",
        "kruskal", "dag", "scc", "topological", "mst", "bridges", "articulation",
    }
    dp_keywords = {"lcs", "matrix_chain", "optimal_bst", "knapsack", "activity"}
    greedy_keywords = {"huffman", "activity_selector", "greedy"}
    string_keywords = {"string", "kmp", "naive_string", "lcs"}
    geometry_keywords = {"convex", "hull", "segment", "graham", "jarvis"}

    if any(kw in name_lower for kw in sort_keywords):
        return ConceptType.SORTING
    if any(kw in name_lower for kw in graph_keywords):
        return ConceptType.GRAPH_TRAVERSAL
    if any(kw in name_lower for kw in dp_keywords):
        return ConceptType.DYNAMIC_PROGRAMMING
    if any(kw in name_lower for kw in greedy_keywords):
        return ConceptType.GREEDY
    if any(kw in name_lower for kw in search_keywords):
        return ConceptType.SEARCHING
    if any(kw in name_lower for kw in string_keywords):
        return ConceptType.STRING_MATCHING
    if any(kw in name_lower for kw in geometry_keywords):
        return ConceptType.GEOMETRY
    return ConceptType.CUSTOM
