"""Derive architect primitives from configured atom source registrations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import importlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, get_args, get_origin

from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.models import AlgorithmicPrimitive, ConceptType, IOSpec
from ageom.sources import AtomSource, SourcesConfig, discover_cdgs, import_atoms, load_sources, resolve_source


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
        inputs.append(
            IOSpec(name=param.name, type_desc=_annotation_to_text(param.annotation))
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
    in_sig = ", ".join(f"{port.name}: {port.type_desc}" for port in inputs) or "void"
    if len(outputs) == 1:
        out_sig = outputs[0].type_desc
    else:
        out_sig = "tuple[" + ", ".join(port.type_desc for port in outputs) + "]"
    return f"({in_sig}) -> {out_sig}"


def _iter_registry_modules() -> Iterable[Any]:
    for module in list(sys.modules.values()):
        if module is None:
            continue
        if hasattr(module, "REGISTRY") and isinstance(getattr(module, "REGISTRY"), dict):
            yield module


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


def _load_atomic_node_index(source: AtomSource, *, base_dir: Path | None = None) -> dict[str, list[_AtomicNodeMeta]]:
    index: dict[str, list[_AtomicNodeMeta]] = defaultdict(list)
    for cdg_path in discover_cdgs(source, base_dir):
        try:
            with open(cdg_path) as handle:
                payload = json.load(handle)
        except Exception:
            continue
        for raw in payload.get("nodes", []) or []:
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
    return index


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
) -> tuple[AlgorithmicPrimitive, list[str]] | None:
    impl = meta.get("impl")
    witness = meta.get("witness")
    module_name = getattr(impl, "__module__", "") or getattr(witness, "__module__", "")
    description = str(meta.get("doc", "") or getattr(impl, "__doc__", "") or "").strip()

    matched = _match_atomic_meta(name, impl, atom_index)
    if matched is not None:
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

    signature_source = witness or impl
    inputs, outputs = _ports_from_callable(signature_source)
    primitive = AlgorithmicPrimitive(
        name=name,
        source=source.name,
        category=_infer_concept_type(name=name, module=module_name, description=description),
        description=description or f"Registered atom from {module_name or source.package}",
        inputs=inputs,
        outputs=outputs,
        type_signature=_signature_from_ports(inputs, outputs),
    )
    aliases = [alias for alias in {getattr(impl, "__name__", ""), name.split(".")[-1]} if alias and alias != name]
    return primitive, aliases


def seed_catalog_from_sources(
    catalog: PrimitiveCatalog,
    *,
    config: SourcesConfig | None = None,
    base_dir: Path | None = None,
) -> int:
    """Add architect primitives derived from configured source registrations."""
    if config is None:
        config = load_sources()

    added = 0
    for source in config.sources:
        root = resolve_source(source, base_dir).expanduser().resolve()
        package_root = root.joinpath(*source.package.split("."))
        import_atoms(source, base_dir)
        try:
            importlib.import_module(f"{source.package}.ghost.registry")
        except Exception:
            pass

        atom_index = _load_atomic_node_index(source, base_dir=base_dir)
        seen_names: set[str] = set()
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
                built = _registration_to_primitive(source, name, meta, atom_index)
                if built is None:
                    continue
                primitive, aliases = built
                existed = catalog.get(primitive.name) is not None
                catalog.add(primitive)
                for alias in aliases:
                    catalog.add_alias(alias, primitive.name)
                seen_names.add(name)
                if not existed:
                    added += 1
    return added
