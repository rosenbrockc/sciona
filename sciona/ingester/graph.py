"""Top-level ingester state machine and IngesterAgent wrapper.

Combines Phase 1 (AST extraction), Phase 2 (semantic chunking),
and Phase 3 (code generation) with verification/repair loops for
mypy type-checking and ghost simulation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sciona.json_utils import extract_json

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from sciona.architect.handoff import CDGExport
from sciona.architect.models import ConceptType, DependencyEdge
from sciona.hunter.llm import LLMClient
from sciona.ingester.chunker import ChunkerDeps, build_chunker_graph
from sciona.ingester.cache import (
    compute_ingest_cache_key,
    load_ingest_cache,
    save_ingest_cache,
)
from sciona.ingester.base_extractor import EXTENSION_MAP, BaseExtractor, SourceLanguage
from sciona.ingester.deterministic_ghost_fixer import build_deterministic_ghost_fixes
from sciona.ingester.deterministic_type_fixer import build_deterministic_type_fixes
from sciona.ingester.emitter import (
    build_procedural_plan,
    emit_ingestion_bundle,
    generate_opaque_witnesses,
)
from sciona.ingester.monitor import IngestMonitor
from sciona.ingester.python_extractor import JAXprExtractor, PythonASTExtractor
from sciona.ingester.treesitter_extractor import TreeSitterExtractor
from sciona.ingester.models import (
    IngestionBundle,
    ProposedMacroPlan,
    RawDataFlowGraph,
    ValidatedMacroPlan,
)
from sciona.ingester.verification_classifier import (
    classify_ghost_failure,
    classify_type_failure,
)
from sciona.ingester.prompts import (
    FIX_MESSAGE_CYCLE_SYSTEM,
    FIX_MESSAGE_CYCLE_USER,
)
from sciona.llm_router import (
    INGESTER_FIX_MESSAGE_CYCLE,
    select_llm,
)
from sciona.shared_context import SharedContextMetrics, SharedContextStore
from sciona.protocols import ProofEnvironment, SemanticIndex

logger = logging.getLogger(__name__)

_MAX_REPAIR_RETRIES = 3

_CONJUGATE_METHOD_HINTS: frozenset[str] = frozenset(
    {
        "fit",
        "posterior",
        "posterior_update",
        "update_hyperparameters",
        "update_params",
        "update_alpha",
        "update_beta",
        "update_conjugate",
    }
)
_SUFF_STAT_HINTS: frozenset[str] = frozenset(
    {
        "sufficient",
        "stat",
        "sum",
        "mean",
        "count",
        "accumulate",
        "aggregate",
        "n_obs",
        "n_samples",
        "successes",
        "failures",
    }
)
_DISTRIBUTION_HINTS: frozenset[str] = frozenset(
    {
        "beta",
        "bernoulli",
        "binomial",
        "dirichlet",
        "categorical",
        "multinomial",
        "gamma",
        "poisson",
        "normal",
        "gaussian",
        "wishart",
        "student",
        "distribution",
        "prior",
        "posterior",
        "hyperparameter",
        "alpha",
        "theta",
    }
)
_DATA_EDGE_HINTS: frozenset[str] = frozenset(
    {
        "data",
        "observation",
        "sample",
        "batch",
        "stats",
        "count",
    }
)
_DIST_EDGE_HINTS: frozenset[str] = frozenset(
    {
        "posterior",
        "distribution",
        "prior",
        "params",
        "hyper",
        "alpha",
        "beta",
    }
)

_BUNDLE_FILE_ATTRS: dict[str, str] = {
    "atoms.py": "generated_atoms",
    "state_models.py": "generated_state_models",
    "witnesses.py": "generated_witnesses",
}


# ---------------------------------------------------------------------------
# State & deps
# ---------------------------------------------------------------------------


class IngesterState(TypedDict):
    source_path: str
    class_name: str
    raw_dfg: RawDataFlowGraph
    validated_plan: ValidatedMacroPlan
    bundle: IngestionBundle
    is_opaque: bool
    mypy_passed: bool
    ghost_passed: bool
    mypy_errors: str
    ghost_errors: str
    type_failure_classification: dict[str, Any]
    ghost_failure_classification: dict[str, Any]
    type_repair_count: int
    ghost_repair_count: int
    done: bool
    error: str


@dataclass
class IngesterDeps:
    llm: LLMClient
    proof_env: ProofEnvironment | None = None
    faiss_index: SemanticIndex | None = None
    output_dir: str | None = None
    max_depth: int = 1
    line_threshold: int = 30
    monitor: IngestMonitor | None = None
    shared_context: SharedContextStore | None = None
    shared_context_metrics: SharedContextMetrics | None = None
    context_namespace: str = ""
    context_budget_chars: int = 900
    parallelism: int = 1


# ---------------------------------------------------------------------------
# Extractor dispatch
# ---------------------------------------------------------------------------


def _has_jax_import(source_path: str) -> bool:
    """Quick check whether a Python file imports JAX."""
    try:
        text = Path(source_path).read_text(errors="replace")
        # Match `import jax`, `from jax import ...`, `import jax.numpy`
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith(("import jax", "from jax ")):
                return True
    except OSError:
        pass
    return False


def _get_monitor(config: RunnableConfig) -> IngestMonitor | None:
    deps = config.get("configurable", {}).get("deps")
    if deps is None:
        return None
    return getattr(deps, "monitor", None)


def _emit_cache_trace(
    monitor: IngestMonitor | None,
    *,
    event_type: str,
    state: str,
    cache_key: str = "",
    cache_path: Path | None = None,
) -> None:
    if monitor is None:
        return
    payload: dict[str, Any] = {"cache": {"state": state}}
    if cache_key:
        payload["cache"]["key"] = cache_key
    if cache_path is not None:
        payload["cache"]["path"] = str(cache_path)
    monitor.trace_event(
        "ingester",
        "cache",
        event_type,
        payload=payload,
    )


def _get_extractor(source_path: str) -> BaseExtractor:
    """Return the appropriate extractor for *source_path* based on extension.

    For ``.py`` files that ``import jax``, returns a ``JAXprExtractor`` that
    enriches the standard AST graph with jaxpr-level metadata (frozen
    stochasticity tracking, grad-target oracle detection).
    """
    ext = Path(source_path).suffix.lower()
    lang = EXTENSION_MAP.get(ext, SourceLanguage.PYTHON)
    if lang == SourceLanguage.PYTHON:
        if _has_jax_import(source_path):
            return JAXprExtractor()
        return PythonASTExtractor()
    return TreeSitterExtractor(lang)


def _bundle_files(bundle: IngestionBundle) -> dict[str, str]:
    """Return non-empty generated bundle files keyed by their module filenames."""
    files: dict[str, str] = {}
    for filename, attr in _BUNDLE_FILE_ATTRS.items():
        source = getattr(bundle, attr, "")
        if source:
            files[filename] = source
    return files


def _normalize_bundle_filename(filename: str | None) -> str:
    if not filename:
        return "atoms.py"
    return Path(filename).name or "atoms.py"


def _apply_line_fixes(source: str, fixes: list[dict[str, Any]]) -> str:
    lines = source.splitlines()
    ordered = sorted(
        fixes,
        key=lambda fix: int(fix.get("line_start", 0) or 0),
        reverse=True,
    )
    for fix in ordered:
        start = max(0, int(fix.get("line_start", 1) or 1) - 1)
        end = max(start, int(fix.get("line_end", start + 1) or (start + 1)))
        replacement = str(fix.get("replacement", "") or "")
        lines[start:end] = replacement.splitlines()
    return "\n".join(lines)


def _apply_bundle_fixes(
    bundle: IngestionBundle, fixes: list[dict[str, Any]]
) -> IngestionBundle | None:
    if not fixes:
        return None

    grouped: dict[str, list[dict[str, Any]]] = {}
    for fix in fixes:
        filename = _normalize_bundle_filename(fix.get("file"))
        grouped.setdefault(filename, []).append(fix)

    updates: dict[str, str] = {}
    for filename, file_fixes in grouped.items():
        attr = _BUNDLE_FILE_ATTRS.get(filename)
        if attr is None:
            continue
        source = getattr(bundle, attr, "")
        if not source:
            continue
        updates[attr] = _apply_line_fixes(source, file_fixes)

    if not updates:
        return None
    return bundle.model_copy(update=updates)


def _publish_verification_failure_snapshot(
    monitor: IngestMonitor | None,
    *,
    state: IngesterState,
    bundle: IngestionBundle,
    stage: str,
    verification_errors: str,
    classification: dict[str, Any],
) -> None:
    """Publish emitted artifacts at a verification failure boundary."""
    if monitor is None:
        return
    try:
        if bundle.generated_atoms:
            monitor.stage_file("debug_atoms.py", bundle.generated_atoms)
        if bundle.generated_state_models:
            monitor.stage_file("debug_state_models.py", bundle.generated_state_models)
        if bundle.generated_witnesses:
            monitor.stage_file("debug_witnesses.py", bundle.generated_witnesses)
        raw_dfg = state.get("raw_dfg")
        if raw_dfg is not None:
            monitor.stage_json("debug_raw_dfg.json", raw_dfg.model_dump(mode="json"))
        validated_plan = state.get("validated_plan")
        if validated_plan is not None:
            monitor.stage_json(
                "debug_validated_plan.json",
                validated_plan.model_dump(mode="json"),
            )
            canonical_ir = getattr(validated_plan.plan, "canonical_ir", None)
            if canonical_ir is not None:
                monitor.stage_json(
                    "debug_ingest_ir.json",
                    canonical_ir.model_dump(mode="json"),
                )
            planning_graph = getattr(validated_plan.plan, "planning_graph", None)
            if planning_graph is not None:
                monitor.stage_json(
                    "debug_planning_graph.json",
                    planning_graph.model_dump(mode="json"),
                )
        monitor.stage_json(
            "debug_typecheck_state.json",
            {
                "stage": stage,
                "mypy_passed": bool(state.get("mypy_passed", False)),
                "ghost_passed": bool(state.get("ghost_passed", False)),
                "type_repair_count": int(state.get("type_repair_count", 0) or 0),
                "ghost_repair_count": int(state.get("ghost_repair_count", 0) or 0),
                "mypy_errors": (
                    verification_errors if stage == "verify_types" else state.get("mypy_errors", "")
                ),
                "ghost_errors": (
                    verification_errors if stage == "verify_ghost" else state.get("ghost_errors", "")
                ),
                "type_failure_classification": (
                    classification
                    if stage == "verify_types"
                    else state.get("type_failure_classification", {})
                ),
                "ghost_failure_classification": (
                    classification
                    if stage == "verify_ghost"
                    else state.get("ghost_failure_classification", {})
                ),
            },
        )
        monitor.stage_json(
            "verification_failure.json",
            {
                "stage": stage,
                "classifier_result": classification,
                "repairable": bool(classification.get("repairable", False)),
                "reason_code": classification.get(
                    "reason_code", "unknown_or_unclassified"
                ),
                "retry_counts": {
                    "type_repair_count": int(state.get("type_repair_count", 0) or 0),
                    "ghost_repair_count": int(state.get("ghost_repair_count", 0) or 0),
                },
                "verification_errors": verification_errors,
            },
        )
        monitor.publish_staged()
    except Exception as exc:
        logger.warning("Failed to publish verification failure snapshot: %s", exc)


# ---------------------------------------------------------------------------
# Conjugate heuristics
# ---------------------------------------------------------------------------


def _atom_id(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _edge_key(e: DependencyEdge) -> tuple[str, str, str, str, str, str]:
    return (
        e.source_id,
        e.target_id,
        e.output_name,
        e.input_name,
        e.source_type,
        e.target_type,
    )


def _contains_hint(text: str, hints: frozenset[str]) -> bool:
    lower = text.lower()
    return any(h in lower for h in hints)


def _method_has_sufficient_stats(mf) -> bool:
    """Heuristic for sufficient-stat accumulation methods."""
    name = mf.name.lower()
    src = (mf.source_code or "").lower()
    if _contains_hint(name, _SUFF_STAT_HINTS):
        return True
    if _contains_hint(src, _SUFF_STAT_HINTS):
        return True
    # Common aggregation patterns in closed-form updates
    patterns = ("sum(", ".sum(", "mean(", ".mean(", "count(", "len(", "+=", "for ")
    return any(p in src for p in patterns)


def _method_has_distribution_context(mf) -> bool:
    text = " ".join(
        [
            mf.name or "",
            mf.return_type or "",
            mf.docstring or "",
            mf.source_code or "",
        ]
    )
    return _contains_hint(text, _DISTRIBUTION_HINTS)


def _method_is_fit_or_posterior(mf) -> bool:
    if _contains_hint(mf.name, _CONJUGATE_METHOD_HINTS):
        return True
    return any(_contains_hint(c, _CONJUGATE_METHOD_HINTS) for c in mf.calls)


def _choose_output(atom) -> tuple[str, str]:
    if atom.outputs:
        preferred = next(
            (
                o
                for o in atom.outputs
                if _contains_hint(o.name, _DATA_EDGE_HINTS | _DIST_EDGE_HINTS)
            ),
            atom.outputs[0],
        )
        return preferred.name, preferred.type_desc
    return "result", "Any"


def _choose_input(atom, output_name: str) -> tuple[str, str]:
    if atom.inputs:
        for inp in atom.inputs:
            if inp.name == output_name:
                return inp.name, inp.type_desc
        preferred = next(
            (
                i
                for i in atom.inputs
                if _contains_hint(i.name, _DATA_EDGE_HINTS | _DIST_EDGE_HINTS)
            ),
            atom.inputs[0],
        )
        return preferred.name, preferred.type_desc
    return output_name, "Any"


def _select_data_atom(macro_atoms, conjugate_id: str):
    for atom in macro_atoms:
        aid = _atom_id(atom.name)
        if aid == conjugate_id:
            continue
        if _contains_hint(
            atom.name, frozenset({"data", "ingest", "observation", "evidence", "stats"})
        ):
            return atom
        if any(_contains_hint(o.name, _DATA_EDGE_HINTS) for o in atom.outputs):
            return atom
    return None


def _select_distribution_atom(macro_atoms, conjugate_id: str):
    for atom in macro_atoms:
        aid = _atom_id(atom.name)
        if aid == conjugate_id:
            continue
        if atom.concept_type in {
            ConceptType.PRIOR_INIT,
            ConceptType.PRIOR_DISTRIBUTION,
        }:
            return atom
        if _contains_hint(
            atom.name, frozenset({"distribution", "posterior", "construct", "build"})
        ):
            return atom
        if any(_contains_hint(i.name, _DIST_EDGE_HINTS) for i in atom.inputs):
            return atom
    return None


def apply_conjugate_heuristics(
    dfg: RawDataFlowGraph,
    validated: ValidatedMacroPlan,
) -> ValidatedMacroPlan:
    """Promote closed-form posterior logic to CONJUGATE_UPDATE and linearize edges."""
    plan = validated.plan
    if not plan.macro_atoms:
        return validated

    by_method = {m.name: m for m in dfg.methods}
    updated_atoms = []
    conjugate_ids: list[str] = []

    for atom in plan.macro_atoms:
        methods = [by_method[m] for m in atom.method_names if m in by_method]
        has_explicit = any(m.is_conjugate for m in methods)
        has_fit_like = any(_method_is_fit_or_posterior(m) for m in methods)
        has_suff_stats = any(_method_has_sufficient_stats(m) for m in methods)
        has_distribution = any(_method_has_distribution_context(m) for m in methods)

        should_promote = has_explicit or (
            has_fit_like and has_suff_stats and has_distribution
        )
        if should_promote and atom.concept_type != ConceptType.CONJUGATE_UPDATE:
            atom = atom.model_copy(
                update={"concept_type": ConceptType.CONJUGATE_UPDATE}
            )

        if atom.concept_type == ConceptType.CONJUGATE_UPDATE:
            conjugate_ids.append(_atom_id(atom.name))
        updated_atoms.append(atom)

    edges = list(plan.edge_definitions)
    seen = {_edge_key(e) for e in edges}
    atom_by_id = {_atom_id(a.name): a for a in updated_atoms}

    for conj_id in conjugate_ids:
        conj_atom = atom_by_id.get(conj_id)
        if conj_atom is None:
            continue

        incoming_exists = any(e.target_id == conj_id for e in edges)
        if not incoming_exists:
            data_atom = _select_data_atom(updated_atoms, conj_id)
            if data_atom is not None:
                out_name, out_type = _choose_output(data_atom)
                in_name, in_type = _choose_input(conj_atom, out_name)
                edge = DependencyEdge(
                    source_id=_atom_id(data_atom.name),
                    target_id=conj_id,
                    output_name=out_name,
                    input_name=in_name,
                    source_type=out_type,
                    target_type=in_type,
                )
                key = _edge_key(edge)
                if key not in seen:
                    seen.add(key)
                    edges.append(edge)

        outgoing_exists = any(e.source_id == conj_id for e in edges)
        if not outgoing_exists:
            dist_atom = _select_distribution_atom(updated_atoms, conj_id)
            if dist_atom is not None:
                out_name, out_type = _choose_output(conj_atom)
                in_name, in_type = _choose_input(dist_atom, out_name)
                edge = DependencyEdge(
                    source_id=conj_id,
                    target_id=_atom_id(dist_atom.name),
                    output_name=out_name,
                    input_name=in_name,
                    source_type=out_type,
                    target_type=in_type,
                )
                key = _edge_key(edge)
                if key not in seen:
                    seen.add(key)
                    edges.append(edge)

    updated_plan = plan.model_copy(
        update={
            "macro_atoms": updated_atoms,
            "edge_definitions": edges,
        }
    )
    return validated.model_copy(update={"plan": updated_plan})


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


async def phase1_extract(
    state: IngesterState, config: RunnableConfig
) -> dict[str, Any]:
    """Phase 1: Deterministic AST extraction."""
    mon = _get_monitor(config)
    if mon:
        mon.phase_start("phase1_extract", step="extract")
    try:
        extractor = _get_extractor(state["source_path"])
        try:
            dfg = await extractor.extract_class(state["source_path"], state["class_name"])
        except ValueError:
            # Not a class — try as a named function
            try:
                dfg = await extractor.extract_function(
                    state["source_path"], state["class_name"]
                )
            except ValueError:
                raise ValueError(
                    f"'{state['class_name']}' not found as class or function "
                    f"in {state['source_path']}"
                )
        if mon:
            mon.phase_end("phase1_extract", step="ok")
        return {"raw_dfg": dfg, "is_opaque": dfg.is_opaque}
    except (FileNotFoundError, ValueError) as exc:
        if mon:
            mon.fail(error=str(exc), phase="phase1_extract")
        return {"error": str(exc), "done": True}


async def phase2_chunk(state: IngesterState, config: RunnableConfig) -> dict[str, Any]:
    """Phase 2: Run the chunker sub-graph."""
    deps: IngesterDeps = config["configurable"]["deps"]
    mon = _get_monitor(config)
    if mon:
        mon.phase_start("phase2_chunk", step="chunk")

    if state.get("error"):
        return {}

    chunker_deps = ChunkerDeps(
        llm=deps.llm,
        faiss_index=deps.faiss_index,
        max_depth=deps.max_depth,
        line_threshold=deps.line_threshold,
        monitor=deps.monitor,
        shared_context=deps.shared_context,
        shared_context_metrics=deps.shared_context_metrics,
        context_namespace=deps.context_namespace,
        context_budget_chars=deps.context_budget_chars,
        parallelism=deps.parallelism,
    )
    chunker_graph = build_chunker_graph().compile()

    initial_state: dict[str, Any] = {
        "raw_dfg": state["raw_dfg"],
        "proposed_plan": ProposedMacroPlan(),
        "validated_plan": ValidatedMacroPlan(plan=ProposedMacroPlan()),
        "critique_passed": False,
        "critique_reason": "",
        "retry_count": 0,
        "missing_attrs": [],
        "done": False,
    }

    chunker_config = {"configurable": {"deps": chunker_deps}}
    final_state = await chunker_graph.ainvoke(initial_state, config=chunker_config)

    validated = final_state.get("validated_plan")
    if validated is None or not validated.all_attrs_accounted:
        # Best-effort: wrap whatever we got
        plan = final_state.get("proposed_plan", ProposedMacroPlan())
        validated = ValidatedMacroPlan(
            plan=plan,
            all_attrs_accounted=False,
            coverage_report=final_state.get("critique_reason", "best-effort"),
        )

    if mon:
        mon.phase_end("phase2_chunk", step="ok")
    return {"validated_plan": validated}


async def phase2_conjugate_heuristics(
    state: IngesterState, config: RunnableConfig
) -> dict[str, Any]:
    """Deterministic pass: detect closed-form conjugate update logic."""
    mon = _get_monitor(config)
    if mon:
        mon.phase_start("phase2_conjugate_heuristics", step="heuristics")
    if state.get("error"):
        return {}
    validated = state.get("validated_plan")
    dfg = state.get("raw_dfg")
    if validated is None or dfg is None:
        return {}
    updated = apply_conjugate_heuristics(dfg, validated)
    if mon:
        mon.phase_end("phase2_conjugate_heuristics", step="ok")
    return {"validated_plan": updated}


async def phase3_emit(state: IngesterState, config: RunnableConfig) -> dict[str, Any]:
    """Phase 3: Generate code and build IngestionBundle."""
    mon = _get_monitor(config)
    if mon:
        mon.phase_start("phase3_emit", step="emit")
    if state.get("error"):
        return {}

    source_lang = state["raw_dfg"].source_language
    bundle = emit_ingestion_bundle(
        state["validated_plan"],
        state["class_name"],
        state["source_path"],
        source_language=source_lang,
    )

    # If opaque, attempt LLM-drafted witnesses and merge into bundle
    if state.get("is_opaque", False):
        deps: IngesterDeps = config["configurable"]["deps"]
        dfg = state["raw_dfg"]
        try:
            witness_source, witness_names = await generate_opaque_witnesses(
                state["validated_plan"].plan.macro_atoms,
                dfg,
                deps.llm,
                shared_context=deps.shared_context,
                shared_context_metrics=deps.shared_context_metrics,
                context_namespace=deps.context_namespace,
                context_budget_chars=deps.context_budget_chars,
                parallelism=deps.parallelism,
            )
            if witness_source.strip():
                bundle = bundle.model_copy(
                    update={
                        "generated_witnesses": witness_source,
                    }
                )
        except Exception as exc:
            logger.warning("Opaque witness generation failed: %s", exc)

    if mon:
        mon.phase_end("phase3_emit", step="ok")
    return {"bundle": bundle}


async def verify_types(state: IngesterState, config: RunnableConfig) -> dict[str, Any]:
    """Write generated source to temp files and run mypy."""
    deps: IngesterDeps = config["configurable"]["deps"]
    bundle: IngestionBundle = state["bundle"]
    mon = _get_monitor(config)
    if mon:
        mon.phase_start("verify_types", step="mypy")

    if state.get("error"):
        return {}

    if deps.proof_env is None:
        return {
            "mypy_passed": True,
            "mypy_errors": "",
            "type_failure_classification": {},
        }

    bundle_files = _bundle_files(bundle)
    try:
        check_generated_files = getattr(deps.proof_env, "check_generated_files", None)
        supports_multi_file_check = callable(check_generated_files) and hasattr(
            deps.proof_env.__class__, "check_generated_files"
        )
        if supports_multi_file_check:
            ok, output = await check_generated_files(
                bundle_files,
                verify_mode="mypy",
                strict=True,
                ignore_missing_imports=True,
            )
        else:
            ok, output = await deps.proof_env.check_proof(bundle.generated_atoms, "")
        if ok:
            bundle_update = bundle.model_copy(update={"mypy_passed": True})
            if mon:
                mon.phase_end("verify_types", step="passed")
            return {
                "bundle": bundle_update,
                "mypy_passed": True,
                "mypy_errors": "",
                "type_failure_classification": {},
            }
        else:
            classification = classify_type_failure(output, source_files=bundle_files)
            _publish_verification_failure_snapshot(
                mon,
                state=state,
                bundle=bundle,
                stage="verify_types",
                verification_errors=output,
                classification=classification,
            )
            if mon:
                mon.phase_end("verify_types", step="failed")
            return {
                "mypy_passed": False,
                "mypy_errors": output,
                "type_failure_classification": classification,
            }
    except Exception as exc:
        logger.warning("mypy verification failed: %s", exc)
        classification = classify_type_failure(str(exc), source_files=bundle_files)
        _publish_verification_failure_snapshot(
            mon,
            state=state,
            bundle=bundle,
            stage="verify_types",
            verification_errors=str(exc),
            classification=classification,
        )
        if mon:
            mon.phase_end("verify_types", step="error")
        return {
            "mypy_passed": False,
            "mypy_errors": str(exc),
            "type_failure_classification": classification,
        }


async def verify_ghost(state: IngesterState, config: RunnableConfig) -> dict[str, Any]:
    """Run ghost simulation on the generated CDG."""
    bundle: IngestionBundle = state["bundle"]
    mon = _get_monitor(config)
    if mon:
        mon.phase_start("verify_ghost", step="ghost_sim")

    if state.get("error"):
        return {}

    try:
        from sciona.synthesizer.ghost_sim import run_ghost_simulation

        report = run_ghost_simulation(bundle.cdg, bundle.match_results)
        bundle_update = bundle.model_copy(
            update={
                "ghost_sim_passed": report.passed or not report.ran,
                "ghost_sim_report": {
                    "ran": report.ran,
                    "passed": report.passed,
                    "error": report.error,
                    "error_node": report.error_node,
                    "error_function": report.error_function,
                    "coverage": report.coverage,
                    "cyclic_deadlock": report.cyclic_deadlock,
                    "deadlock_nodes": report.deadlock_nodes,
                    "iterations_used": report.iterations_used,
                },
            }
        )
        passed = report.passed or not report.ran
        errors = report.error if not passed else ""
        classification: dict[str, Any] = {}
        if not passed:
            classification = classify_ghost_failure(
                bundle_update.ghost_sim_report,
                witness_source=bundle.generated_witnesses,
            )
            _publish_verification_failure_snapshot(
                mon,
                state=state,
                bundle=bundle_update,
                stage="verify_ghost",
                verification_errors=errors,
                classification=classification,
            )
        if mon:
            mon.phase_end("verify_ghost", step="passed" if passed else "failed")
        return {
            "bundle": bundle_update,
            "ghost_passed": passed,
            "ghost_errors": errors,
            "ghost_failure_classification": classification,
        }
    except ImportError:
        if mon:
            mon.phase_end("verify_ghost", step="skipped")
        return {
            "ghost_passed": True,
            "ghost_errors": "",
            "ghost_failure_classification": {},
        }


async def repair_types(state: IngesterState, config: RunnableConfig) -> dict[str, Any]:
    """Apply deterministic repairs for narrow mypy failures."""
    bundle: IngestionBundle = state["bundle"]
    mon = _get_monitor(config)
    if mon:
        mon.phase_start("repair_types", step="deterministic_fix_type")

    classification = state.get("type_failure_classification") or classify_type_failure(
        state.get("mypy_errors", ""),
        source_files=_bundle_files(bundle),
    )
    if not classification.get("repairable") or classification.get("route") != "repair_types":
        if mon:
            mon.phase_end("repair_types", step="blocked")
        return {}

    fixes = build_deterministic_type_fixes(
        state.get("mypy_errors", ""),
        _bundle_files(bundle),
    )
    if fixes:
        updated_bundle = _apply_bundle_fixes(bundle, fixes)
        if updated_bundle is not None:
            if mon:
                mon.phase_end("repair_types", step="patched")
            return {
                "bundle": updated_bundle,
                "type_repair_count": state.get("type_repair_count", 0) + 1,
            }

    if mon:
        mon.phase_end("repair_types", step="done")
    return {"type_repair_count": state.get("type_repair_count", 0) + 1}


async def repair_ghost(state: IngesterState, config: RunnableConfig) -> dict[str, Any]:
    """Apply deterministic witness repairs for narrow ghost failures."""
    bundle: IngestionBundle = state["bundle"]
    report = bundle.ghost_sim_report
    mon = _get_monitor(config)
    if mon:
        mon.phase_start("repair_ghost", step="deterministic_fix_ghost")

    classification = state.get("ghost_failure_classification") or classify_ghost_failure(
        report,
        witness_source=bundle.generated_witnesses,
    )
    if not classification.get("repairable") or classification.get("route") != "repair_ghost":
        if mon:
            mon.phase_end("repair_ghost", step="blocked")
        return {}

    fixes = build_deterministic_ghost_fixes(
        str(report.get("error_node", "") or ""),
        str(report.get("error_function", "") or ""),
        str(report.get("error", "") or ""),
        bundle.generated_witnesses,
    )
    if fixes:
        updated_witnesses = bundle.generated_witnesses
        for fix in fixes:
            replacement = fix.get("replacement", "")
            if replacement:
                updated_witnesses = replacement
                break
        updated_bundle = bundle.model_copy(
            update={
                "generated_witnesses": updated_witnesses,
            }
        )
        if mon:
            mon.phase_end("repair_ghost", step="patched")
        return {
            "bundle": updated_bundle,
            "ghost_repair_count": state.get("ghost_repair_count", 0) + 1,
        }

    if mon:
        mon.phase_end("repair_ghost", step="done")
    return {"ghost_repair_count": state.get("ghost_repair_count", 0) + 1}


async def repair_message_cycle(
    state: IngesterState, config: RunnableConfig
) -> dict[str, Any]:
    """LLM-assisted repair of cyclic deadlocks in message-passing topologies."""
    deps: IngesterDeps = config["configurable"]["deps"]
    bundle: IngestionBundle = state["bundle"]
    report = bundle.ghost_sim_report
    mon = _get_monitor(config)
    if mon:
        mon.phase_start("repair_message_cycle", step="llm_fix_message_cycle")

    # Collect cycle edges for the prompt
    deadlock_nodes = report.get("deadlock_nodes", [])
    cycle_edges = []
    for edge in bundle.cdg.edges:
        if edge.source_id in deadlock_nodes or edge.target_id in deadlock_nodes:
            cycle_edges.append(f"{edge.source_id} -> {edge.target_id}")

    user_prompt = FIX_MESSAGE_CYCLE_USER.format(
        deadlock_nodes=", ".join(deadlock_nodes),
        cycle_edges="\n".join(cycle_edges),
        witness_source=bundle.generated_witnesses,
    )

    if mon:
        mon.llm_start(INGESTER_FIX_MESSAGE_CYCLE)
    try:
        response = await select_llm(deps.llm, INGESTER_FIX_MESSAGE_CYCLE).complete(
            FIX_MESSAGE_CYCLE_SYSTEM, user_prompt
        )
        if mon:
            mon.llm_end(INGESTER_FIX_MESSAGE_CYCLE, ok=True)
    except Exception as exc:
        if mon:
            mon.llm_end(INGESTER_FIX_MESSAGE_CYCLE, ok=False, error=str(exc))
        raise

    try:
        fixes = extract_json(response)
        if isinstance(fixes, list) and fixes:
            lines = bundle.generated_witnesses.splitlines()
            for fix in sorted(
                fixes, key=lambda f: f.get("line_start", 0), reverse=True
            ):
                start = fix.get("line_start", 1) - 1
                end = fix.get("line_end", start + 1)
                replacement = fix.get("replacement", "")
                lines[start:end] = replacement.splitlines()
            updated_witnesses = "\n".join(lines)
            updated_bundle = bundle.model_copy(
                update={
                    "generated_witnesses": updated_witnesses,
                }
            )
            if mon:
                mon.phase_end("repair_message_cycle", step="patched")
            return {
                "bundle": updated_bundle,
                "ghost_repair_count": state.get("ghost_repair_count", 0) + 1,
            }
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Failed to parse message cycle repair response: %s", exc)

    if mon:
        mon.phase_end("repair_message_cycle", step="done")
    return {"ghost_repair_count": state.get("ghost_repair_count", 0) + 1}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_type_check(state: IngesterState) -> str:
    if state.get("mypy_passed", False):
        return "verify_ghost"
    classification = state.get("type_failure_classification") or classify_type_failure(
        state.get("mypy_errors", ""),
        source_files=_bundle_files(state["bundle"]) if state.get("bundle") else {},
    )
    if (
        state.get("type_repair_count", 0) >= _MAX_REPAIR_RETRIES
        or not classification.get("repairable")
        or classification.get("route") != "repair_types"
    ):
        return "end"
    return "repair_types"


def route_after_ghost(state: IngesterState) -> str:
    if state.get("ghost_passed", False):
        return "end"
    bundle: IngestionBundle | None = state.get("bundle")
    report = getattr(bundle, "ghost_sim_report", None) or {}
    classification = state.get("ghost_failure_classification") or classify_ghost_failure(
        report,
        witness_source=bundle.generated_witnesses if bundle is not None else "",
    )
    if (
        state.get("ghost_repair_count", 0) >= _MAX_REPAIR_RETRIES
        or not classification.get("repairable")
    ):
        return "end"
    route = classification.get("route")
    if route == "repair_message_cycle":
        return "repair_message_cycle"
    if route == "repair_ghost":
        return "repair_ghost"
    return "end"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_ingester_graph() -> StateGraph:
    """Construct the top-level ingester state machine."""
    graph = StateGraph(IngesterState)

    graph.add_node("phase1_extract", phase1_extract)
    graph.add_node("phase2_chunk", phase2_chunk)
    graph.add_node("phase2_conjugate_heuristics", phase2_conjugate_heuristics)
    graph.add_node("phase3_emit", phase3_emit)
    graph.add_node("verify_types", verify_types)
    graph.add_node("verify_ghost", verify_ghost)
    graph.add_node("repair_types", repair_types)
    graph.add_node("repair_ghost", repair_ghost)
    graph.add_node("repair_message_cycle", repair_message_cycle)

    graph.set_entry_point("phase1_extract")
    graph.add_edge("phase1_extract", "phase2_chunk")
    graph.add_edge("phase2_chunk", "phase2_conjugate_heuristics")
    graph.add_edge("phase2_conjugate_heuristics", "phase3_emit")
    graph.add_edge("phase3_emit", "verify_types")

    graph.add_conditional_edges(
        "verify_types",
        route_after_type_check,
        {
            "verify_ghost": "verify_ghost",
            "repair_types": "repair_types",
            "end": END,
        },
    )
    graph.add_edge("repair_types", "verify_types")

    graph.add_conditional_edges(
        "verify_ghost",
        route_after_ghost,
        {
            "end": END,
            "repair_ghost": "repair_ghost",
            "repair_message_cycle": "repair_message_cycle",
        },
    )
    graph.add_edge("repair_ghost", "verify_types")
    graph.add_edge("repair_message_cycle", "verify_types")

    return graph


# ---------------------------------------------------------------------------
# Agent wrapper
# ---------------------------------------------------------------------------


class IngesterAgent:
    """High-level wrapper for the Smart Ingester pipeline.

    Usage::

        agent = IngesterAgent(llm=llm, proof_env=env)
        bundle = await agent.ingest("path/to/file.py", "ClassName")
    """

    def __init__(
        self,
        llm: LLMClient,
        proof_env: ProofEnvironment | None = None,
        faiss_index: SemanticIndex | None = None,
        output_dir: str | None = None,
        max_repair_retries: int = 3,
        max_depth: int = 1,
        line_threshold: int = 30,
        monitor: IngestMonitor | None = None,
        shared_context: SharedContextStore | None = None,
        shared_context_metrics: SharedContextMetrics | None = None,
        context_namespace: str = "",
        context_budget_chars: int = 900,
        parallelism: int = 1,
        enable_cache: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        global _MAX_REPAIR_RETRIES
        _MAX_REPAIR_RETRIES = max_repair_retries

        self._deps = IngesterDeps(
            llm=llm,
            proof_env=proof_env,
            faiss_index=faiss_index,
            output_dir=output_dir,
            max_depth=max_depth,
            line_threshold=line_threshold,
            monitor=monitor,
            shared_context=shared_context,
            shared_context_metrics=shared_context_metrics,
            context_namespace=context_namespace,
            context_budget_chars=context_budget_chars,
            parallelism=max(1, parallelism),
        )
        self._graph = build_ingester_graph().compile()
        self._enable_cache = enable_cache
        self._cache_dir = Path(cache_dir) if cache_dir else Path("data/ingest_cache")

    async def ingest(
        self, source_path: str, class_name: str, *, raise_on_error: bool = False
    ) -> IngestionBundle:
        """Run the full ingester pipeline.

        Args:
            source_path: Path to the Python source file.
            class_name: Name of the class to ingest.

        Returns:
            An ``IngestionBundle`` with CDG, generated source, and match results.
        """
        cache_key = ""
        monitor = self._deps.monitor
        if self._enable_cache:
            try:
                cache_key = compute_ingest_cache_key(
                    source_path=source_path,
                    class_name=class_name,
                    max_depth=self._deps.max_depth,
                    line_threshold=self._deps.line_threshold,
                )
                cached = load_ingest_cache(self._cache_dir, cache_key)
                if cached is not None:
                    _emit_cache_trace(
                        monitor,
                        event_type="CACHE_LOOKUP",
                        state="hit",
                        cache_key=cache_key,
                    )
                    return cached
                _emit_cache_trace(
                    monitor,
                    event_type="CACHE_LOOKUP",
                    state="miss",
                    cache_key=cache_key,
                )
            except Exception:
                cache_key = ""

        final_state = await self.ingest_state(source_path, class_name)
        if raise_on_error and final_state.get("error"):
            raise RuntimeError(str(final_state["error"]))

        bundle: IngestionBundle = final_state["bundle"]
        if self._enable_cache and cache_key and not final_state.get("error"):
            try:
                cache_path = save_ingest_cache(self._cache_dir, cache_key, bundle)
                _emit_cache_trace(
                    monitor,
                    event_type="CACHE_STORE",
                    state="miss",
                    cache_key=cache_key,
                    cache_path=cache_path,
                )
            except Exception:
                pass
        return bundle

    async def ingest_state(self, source_path: str, class_name: str) -> dict[str, Any]:
        """Run the full ingester pipeline and return the terminal graph state."""
        initial_state: dict[str, Any] = {
            "source_path": source_path,
            "class_name": class_name,
            "raw_dfg": RawDataFlowGraph(class_name=class_name),
            "validated_plan": ValidatedMacroPlan(plan=ProposedMacroPlan()),
            "bundle": IngestionBundle(
                cdg=CDGExport(nodes=[], edges=[]),
            ),
            "is_opaque": False,
            "mypy_passed": False,
            "ghost_passed": False,
            "mypy_errors": "",
            "ghost_errors": "",
            "type_failure_classification": {},
            "ghost_failure_classification": {},
            "type_repair_count": 0,
            "ghost_repair_count": 0,
            "done": False,
            "error": "",
        }

        config: dict[str, Any] = {"configurable": {"deps": self._deps}}
        return await self._graph.ainvoke(initial_state, config=config)

    async def ingest_procedural(
        self, source_path: str, pipeline_name: str | None = None
    ) -> IngestionBundle:
        """Ingest a procedural Python script using SSA edge inference.

        No LLM calls — edges are determined deterministically from variable
        tracking.  Skips Phase 2 (chunker) and verification loops entirely.

        Args:
            source_path: Path to the Python source file.
            pipeline_name: Display name for the pipeline (defaults to file stem).

        Returns:
            An ``IngestionBundle`` with CDG, generated source, and match results.
        """
        name = pipeline_name or Path(source_path).stem
        mon = self._deps.monitor
        if mon:
            mon.phase_start("procedural_extract", step="extract")
        extractor = _get_extractor(source_path)
        dfg = await extractor.extract_procedural(source_path, name)
        if mon:
            mon.phase_end("procedural_extract", step="ok")
            mon.phase_start("procedural_emit", step="emit")
        plan = build_procedural_plan(dfg, name)
        bundle = emit_ingestion_bundle(
            plan,
            name,
            source_path,
            source_language=dfg.source_language,
        )
        if mon:
            mon.phase_end("procedural_emit", step="ok")
        return bundle
