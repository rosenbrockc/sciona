"""Derive architect primitives from configured atom source registrations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import ast
import importlib
import inspect
import json
import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, get_args, get_origin

from ageom.architect.catalog import CatalogReport, PrimitiveCatalog
from ageom.architect.models import AlgorithmicPrimitive, ConceptType, IOSpec
from ageom.sources import AtomSource, SourcesConfig, discover_cdgs, import_atoms, load_sources, resolve_source

if TYPE_CHECKING:
    from ageom.architect.embedder import SkillIndex

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


@dataclass
class _AtomicNodeMeta:
    node_id: str
    name: str
    description: str
    concept_type: ConceptType
    inputs: list[IOSpec]
    outputs: list[IOSpec]
    type_signature: str
    matched_primitive: str


def _annotation_to_text(annotation: Any) -> str:
    if annotation is inspect._empty:
        return "Any"
    if annotation is None:
        return "None"
    if isinstance(annotation, str):
        return annotation
    if getattr(annotation, "__module__", "") == "builtins":
        return getattr(annotation, "__name__", repr(annotation))
    rendered = repr(annotation)
    for prefix in ("typing.", "collections.abc."):
        if rendered.startswith(prefix):
            rendered = rendered[len(prefix) :]
    return rendered


def _ports_from_callable(func: Any) -> tuple[list[IOSpec], list[IOSpec]]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return [], [IOSpec(name="result", type_desc="Any")]

    inputs: list[IOSpec] = []
    for param in signature.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.name in {"self", "cls"}:
            continue
        required = param.default is inspect._empty
        default_value_repr = "" if required else repr(param.default)
        inputs.append(
            IOSpec(
                name=param.name,
                type_desc=_annotation_to_text(param.annotation),
                required=required,
                default_value_repr=default_value_repr,
            )
        )

    return_annotation = signature.return_annotation
    origin = get_origin(return_annotation)
    args = get_args(return_annotation)
    if origin is tuple and args and args[-1] is not Ellipsis:
        outputs = [
            IOSpec(name=f"result_{idx + 1}", type_desc=_annotation_to_text(arg))
            for idx, arg in enumerate(args)
        ]
    else:
        outputs = [IOSpec(name="result", type_desc=_annotation_to_text(return_annotation))]
    return inputs, outputs


def _signature_from_ports(inputs: list[IOSpec], outputs: list[IOSpec]) -> str:
    in_sig = ", ".join(_format_port_signature(port) for port in inputs) or "void"
    if len(outputs) == 1:
        out_sig = outputs[0].type_desc
    else:
        out_sig = "tuple[" + ", ".join(port.type_desc for port in outputs) + "]"
    return f"({in_sig}) -> {out_sig}"


def _format_port_signature(port: IOSpec) -> str:
    rendered = f"{port.name}: {port.type_desc}"
    if not port.required:
        rendered += "?"
        if port.default_value_repr:
            rendered += f" = {port.default_value_repr}"
    return rendered


def _iter_registry_modules() -> Iterable[Any]:
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            registry = getattr(module, "REGISTRY")
        except Exception:
            continue
        if isinstance(registry, dict):
            yield module


def _package_python_files(package_root: Path) -> list[Path]:
    return sorted(path for path in package_root.rglob("*.py") if path.name != "__init__.py")


def _module_belongs_to_source(
    module_name: str,
    path: str | None,
    *,
    source: AtomSource,
    package_root: Path,
) -> bool:
    if module_name == source.package or module_name.startswith(source.package + "."):
        return True
    if path:
        try:
            return Path(path).resolve().is_relative_to(package_root)
        except Exception:
            return False
    return False


def _load_atomic_node_index(
    source: AtomSource, *, base_dir: Path | None = None
) -> tuple[dict[str, list[_AtomicNodeMeta]], dict[str, str]]:
    """Load atomic node metadata and topo-hashes from CDG files.

    Returns ``(atom_index, topo_hashes)`` where *topo_hashes* maps
    ``topo_hash -> source_name`` for structural de-duplication.
    """
    from ageom.graph_store import _topo_hash

    index: dict[str, list[_AtomicNodeMeta]] = defaultdict(list)
    topo_hashes: dict[str, str] = {}

    for cdg_path in discover_cdgs(source, base_dir):
        try:
            with open(cdg_path) as handle:
                payload = json.load(handle)
        except Exception:
            continue

        all_nodes_raw = payload.get("nodes", []) or []
        all_edges_raw = payload.get("edges", []) or []

        # Compute topo_hashes for decomposed parent nodes.
        for raw in all_nodes_raw:
            children = raw.get("children", []) or []
            status = raw.get("status", "")
            if status == "decomposed" and len(children) >= 2:
                node_id = str(raw.get("node_id", ""))
                nodes_dicts = [
                    {"node_id": str(n.get("node_id", "")), "parent_id": str(n.get("parent_id", ""))}
                    for n in all_nodes_raw
                ]
                edges_dicts = [
                    {"source_id": str(e.get("source_id", "")), "target_id": str(e.get("target_id", ""))}
                    for e in all_edges_raw
                ]
                try:
                    h = _topo_hash(nodes_dicts, edges_dicts, node_id)
                    if h:
                        topo_hashes[h] = source.name
                except Exception:
                    pass

        for raw in all_nodes_raw:
            if raw.get("status") != "atomic":
                continue
            try:
                concept_type = ConceptType(raw.get("concept_type", ConceptType.CUSTOM.value))
            except ValueError:
                concept_type = ConceptType.CUSTOM
            meta = _AtomicNodeMeta(
                node_id=str(raw.get("node_id", "")),
                name=str(raw.get("name", "")),
                description=str(raw.get("description", "")),
                concept_type=concept_type,
                inputs=[IOSpec.model_validate(item) for item in raw.get("inputs", []) or []],
                outputs=[IOSpec.model_validate(item) for item in raw.get("outputs", []) or []],
                type_signature=str(raw.get("type_signature", "")),
                matched_primitive=str(raw.get("matched_primitive", "") or ""),
            )
            for key in {
                _normalize(meta.node_id),
                _normalize(meta.name),
                _normalize(meta.matched_primitive),
            }:
                if key:
                    index[key].append(meta)
    return index, topo_hashes


def _expr_to_text(expr: ast.expr | None) -> str:
    if expr is None:
        return "Any"
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    try:
        return ast.unparse(expr)
    except Exception:
        return "Any"


def _literal_default_repr(expr: ast.expr | None) -> str:
    if expr is None:
        return ""
    try:
        return ast.unparse(expr)
    except Exception:
        return ""


def _ports_from_ast_function(func: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[list[IOSpec], list[IOSpec]]:
    posonly = list(func.args.posonlyargs)
    regular = list(func.args.args)
    if regular and regular[0].arg in {"self", "cls"}:
        regular = regular[1:]
    ordered = posonly + regular
    defaults = list(func.args.defaults)
    required_prefix = len(ordered) - len(defaults)

    inputs: list[IOSpec] = []
    for idx, arg in enumerate(ordered):
        default_expr = defaults[idx - required_prefix] if idx >= required_prefix else None
        required = default_expr is None
        inputs.append(
            IOSpec(
                name=arg.arg,
                type_desc=_expr_to_text(arg.annotation),
                required=required,
                default_value_repr=_literal_default_repr(default_expr),
            )
        )

    for arg, default_expr in zip(func.args.kwonlyargs, func.args.kw_defaults):
        if arg.arg in {"self", "cls"}:
            continue
        required = default_expr is None
        inputs.append(
            IOSpec(
                name=arg.arg,
                type_desc=_expr_to_text(arg.annotation),
                required=required,
                default_value_repr=_literal_default_repr(default_expr),
            )
        )

    return_annotation = func.returns
    if isinstance(return_annotation, ast.Subscript) and _expr_to_text(return_annotation.value) == "tuple":
        slice_value = return_annotation.slice
        tuple_elts = slice_value.elts if isinstance(slice_value, ast.Tuple) else [slice_value]
        outputs = [
            IOSpec(name=f"result_{idx + 1}", type_desc=_expr_to_text(item))
            for idx, item in enumerate(tuple_elts)
        ]
    else:
        outputs = [IOSpec(name="result", type_desc=_expr_to_text(return_annotation))]
    return inputs, outputs


def _ports_are_uninformative(inputs: list[IOSpec], outputs: list[IOSpec]) -> bool:
    if not inputs:
        return True
    if all(port.type_desc == "Any" for port in inputs) and all(
        port.type_desc == "Any" for port in outputs
    ):
        return True
    return False


_KEYWORD_TYPES: list[tuple[tuple[str, ...], ConceptType]] = [
    (("sort",), ConceptType.SORTING),
    (("search", "lookup"), ConceptType.SEARCHING),
    (("graph", "laplacian", "dijkstra", "shortest_path"), ConceptType.GRAPH_TRAVERSAL),
    (("filter", "sosfilt", "lfilter", "butter", "cheby", "freqz"), ConceptType.SIGNAL_FILTER),
    (("fft", "dct", "spectrum", "spectral"), ConceptType.SIGNAL_TRANSFORM),
    (("particle", "posterior", "prior", "mcmc", "resample"), ConceptType.SAMPLER),
    (("kalman",), ConceptType.CONJUGATE_UPDATE),
    (("state", "initialize", "bootstrap"), ConceptType.STATE_INIT),
    (("assemble", "adapter", "builder"), ConceptType.DATA_ASSEMBLY),
    (("progress", "diagnostic", "plot", "render", "visual"), ConceptType.OBSERVABILITY),
]


def _infer_concept_type(*, name: str, module: str, description: str) -> ConceptType:
    haystack = f"{name} {module} {description}".lower()
    for words, concept_type in _KEYWORD_TYPES:
        if any(word in haystack for word in words):
            return concept_type
    return ConceptType.CUSTOM


def _ast_register_atom_spec(decorator: ast.AST) -> tuple[str, str] | None:
    if isinstance(decorator, ast.Call):
        target = decorator.func
        if isinstance(target, ast.Name) and target.id == "register_atom":
            witness_name = ""
            if decorator.args:
                first = decorator.args[0]
                if isinstance(first, ast.Name):
                    witness_name = first.id
                elif isinstance(first, ast.Attribute):
                    witness_name = first.attr
            for kw in decorator.keywords:
                if kw.arg == "name":
                    try:
                        value = ast.literal_eval(kw.value)
                    except Exception:
                        value = None
                    if isinstance(value, str) and value:
                        return value, witness_name
            return "", witness_name
    elif isinstance(decorator, ast.Name) and decorator.id == "register_atom":
        return "", ""
    return None


def _parse_register_atom_functions(package_root: Path, package_name: str) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for py_file in _package_python_files(package_root):
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except Exception:
            continue
        module_name = ".".join(py_file.relative_to(package_root.parent).with_suffix("").parts)
        function_defs = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            reg_spec = None
            for decorator in node.decorator_list:
                reg_spec = _ast_register_atom_spec(decorator)
                if reg_spec is not None:
                    break
            if reg_spec is None:
                continue
            reg_name, witness_name = reg_spec
            atom_name = reg_name or node.name
            inputs, outputs = _ports_from_ast_function(node)
            doc = ast.get_docstring(node) or ""
            witness_doc = ""
            witness_inputs: list[IOSpec] = []
            witness_outputs: list[IOSpec] = []
            doc_source = "impl"
            signature_source = "impl"
            witness_func = function_defs.get(witness_name)
            if witness_func is not None:
                witness_doc = ast.get_docstring(witness_func) or ""
                witness_inputs, witness_outputs = _ports_from_ast_function(witness_func)
                if not doc and witness_doc:
                    doc_source = "witness"
                if _ports_are_uninformative(inputs, outputs) and not _ports_are_uninformative(
                    witness_inputs, witness_outputs
                ):
                    inputs = witness_inputs
                    outputs = witness_outputs
                    signature_source = "witness"
            parsed[atom_name] = {
                "impl": None,
                "witness": None,
                "doc": doc or witness_doc,
                "doc_source": doc_source if (doc or witness_doc) else "",
                "module": module_name,
                "name": atom_name,
                "ast_inputs": inputs,
                "ast_outputs": outputs,
                "signature_source": signature_source,
                "witness_name": witness_name,
            }
    return parsed


def _match_atomic_meta(
    name: str,
    impl: Any,
    atom_index: dict[str, list[_AtomicNodeMeta]],
) -> _AtomicNodeMeta | None:
    candidates = [
        _normalize(name),
        _normalize(name.split(".")[-1]),
        _normalize(getattr(impl, "__name__", "")),
    ]
    for key in candidates:
        if not key:
            continue
        matches = atom_index.get(key)
        if matches:
            return matches[0]
    return None


def _registration_to_primitive(
    source: AtomSource,
    name: str,
    meta: dict[str, Any],
    atom_index: dict[str, list[_AtomicNodeMeta]],
    *,
    report: CatalogReport | None = None,
) -> tuple[AlgorithmicPrimitive, list[str]] | None:
    impl = meta.get("impl")
    witness = meta.get("witness")
    module_name = getattr(impl, "__module__", "") or getattr(witness, "__module__", "")
    meta_doc = str(meta.get("doc", "") or "").strip()
    impl_doc = str(getattr(impl, "__doc__", "") or "").strip()
    witness_doc = str(getattr(witness, "__doc__", "") or "").strip()
    description = str(meta_doc or impl_doc or witness_doc).strip()
    if report is not None:
        if meta.get("doc_source") == "witness" or (not meta_doc and not impl_doc and witness_doc):
            report.source_witness_doc_fallbacks += 1
        if meta.get("signature_source") == "witness":
            report.source_witness_signature_fallbacks += 1

    matched = _match_atomic_meta(name, impl, atom_index)
    if matched is not None:
        if report is not None:
            report.source_cdg_metadata_matches += 1
        primitive = AlgorithmicPrimitive(
            name=name,
            source=source.name,
            category=matched.concept_type,
            description=matched.description or description or name,
            inputs=matched.inputs,
            outputs=matched.outputs,
            type_signature=matched.type_signature or _signature_from_ports(matched.inputs, matched.outputs),
        )
        aliases = [alias for alias in {matched.node_id, matched.name, matched.matched_primitive} if alias and alias != name]
        return primitive, aliases

    if meta.get("ast_inputs") is not None:
        inputs = meta.get("ast_inputs", [])
        outputs = meta.get("ast_outputs", []) or [IOSpec(name="result", type_desc="Any")]
    else:
        signature_source = witness or impl
        inputs, outputs = _ports_from_callable(signature_source)
    primitive = AlgorithmicPrimitive(
        name=name,
        source=source.name,
        category=_infer_concept_type(
            name=name,
            module=module_name or str(meta.get("module", "")),
            description=description,
        ),
        description=description or f"Registered atom from {module_name or source.package}",
        inputs=inputs,
        outputs=outputs,
        type_signature=_signature_from_ports(inputs, outputs),
    )
    aliases = [alias for alias in {getattr(impl, "__name__", ""), name.split(".")[-1]} if alias and alias != name]
    return primitive, aliases


def _add_primitive_with_aliases(
    catalog: PrimitiveCatalog,
    primitive: AlgorithmicPrimitive,
    aliases: list[str],
    *,
    skill_index: SkillIndex | None = None,
    dedup_threshold: float = 0.85,
    report: CatalogReport | None = None,
) -> bool:
    """Insert *primitive* via dedup, register aliases.  Returns True if added."""
    result = catalog.add_with_dedup(
        primitive, skill_index, dedup_threshold, report
    )
    canonical = result.incumbent_name if result.is_duplicate else primitive.name
    for alias in aliases:
        try:
            catalog.add_alias(alias, canonical)
        except KeyError:
            pass
    if skill_index is not None and not result.is_duplicate:
        skill_index.add_primitive(primitive)
    return not result.is_duplicate


def seed_catalog_from_sources(
    catalog: PrimitiveCatalog,
    *,
    config: SourcesConfig | None = None,
    base_dir: Path | None = None,
    include_live_registries: bool = True,
    skill_index: SkillIndex | None = None,
    dedup_threshold: float = 0.85,
    report: CatalogReport | None = None,
) -> int:
    """Add architect primitives derived from configured source registrations."""
    if config is None:
        config = load_sources()

    added = 0
    seen_topo_hashes: dict[str, str] = {}  # topo_hash -> first source name

    for source in config.sources:
        root = resolve_source(source, base_dir).expanduser().resolve()
        package_root = root.joinpath(*source.package.split("."))
        atom_index, source_topo_hashes = _load_atomic_node_index(
            source, base_dir=base_dir
        )

        # Structural de-duplication: skip CDG subtrees already seen.
        structural_skips: set[str] = set()
        for h, src_name in source_topo_hashes.items():
            if h in seen_topo_hashes and seen_topo_hashes[h] != src_name:
                logger.debug(
                    "Structural duplicate: topo_hash %s from '%s' "
                    "already seen from '%s'",
                    h,
                    src_name,
                    seen_topo_hashes[h],
                )
                structural_skips.add(h)
                if report is not None:
                    report.structural_skips += 1
            else:
                seen_topo_hashes[h] = src_name

        ast_entries = _parse_register_atom_functions(package_root, source.package)
        seen_names: set[str] = set()
        if include_live_registries:
            import_atoms(source, base_dir)
            try:
                importlib.import_module(f"{source.package}.ghost.registry")
            except Exception:
                pass

            for registry_module in _iter_registry_modules():
                registry = getattr(registry_module, "REGISTRY", {})
                for name, meta in registry.items():
                    impl = meta.get("impl")
                    witness = meta.get("witness")
                    module_name = getattr(impl, "__module__", "") or getattr(witness, "__module__", "")
                    file_path = None
                    for fn in (impl, witness):
                        try:
                            if fn is not None:
                                file_path = inspect.getsourcefile(fn) or inspect.getfile(fn)
                                if file_path:
                                    break
                        except Exception:
                            continue
                    if not _module_belongs_to_source(
                        module_name,
                        file_path,
                        source=source,
                        package_root=package_root,
                    ):
                        continue
                    if name in seen_names:
                        continue
                    if report is not None:
                        report.source_live_registry_candidates += 1
                    built = _registration_to_primitive(
                        source,
                        name,
                        meta,
                        atom_index,
                        report=report,
                    )
                    if built is None:
                        continue
                    primitive, aliases = built
                    if _add_primitive_with_aliases(
                        catalog,
                        primitive,
                        aliases,
                        skill_index=skill_index,
                        dedup_threshold=dedup_threshold,
                        report=report,
                    ):
                        added += 1
                    seen_names.add(name)

        for name, meta in ast_entries.items():
            if name in seen_names or catalog.get(name) is not None:
                continue
            if report is not None:
                report.source_ast_candidates += 1
            built = _registration_to_primitive(
                source,
                name,
                meta,
                atom_index,
                report=report,
            )
            if built is None:
                continue
            primitive, aliases = built
            if _add_primitive_with_aliases(
                catalog,
                primitive,
                aliases,
                skill_index=skill_index,
                dedup_threshold=dedup_threshold,
                report=report,
            ):
                added += 1
            seen_names.add(name)
    return added
