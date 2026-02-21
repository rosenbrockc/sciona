"""Top-level ingester state machine and IngesterAgent wrapper.

Combines Phase 1 (AST extraction), Phase 2 (semantic chunking),
and Phase 3 (code generation) with verification/repair loops for
mypy type-checking and ghost simulation.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from ageom.architect.handoff import CDGExport
from ageom.architect.models import ConceptType, DependencyEdge
from ageom.hunter.llm import LLMClient
from ageom.ingester.chunker import ChunkerDeps, ChunkerState, build_chunker_graph
from ageom.ingester.base_extractor import EXTENSION_MAP, BaseExtractor, SourceLanguage
from ageom.ingester.emitter import (
    build_procedural_plan,
    emit_ingestion_bundle,
    generate_opaque_witnesses,
)
from ageom.ingester.extractor import extract_data_flow, extract_procedural_data_flow
from ageom.ingester.python_extractor import JAXprExtractor, PythonASTExtractor
from ageom.ingester.treesitter_extractor import TreeSitterExtractor
from ageom.ingester.models import (
    IngestionBundle,
    ProposedMacroPlan,
    RawDataFlowGraph,
    ValidatedMacroPlan,
)
from ageom.ingester.prompts import (
    FIX_GHOST_ERROR_SYSTEM,
    FIX_GHOST_ERROR_USER,
    FIX_TYPE_ERROR_SYSTEM,
    FIX_TYPE_ERROR_USER,
)
from ageom.protocols import ProofEnvironment, SemanticIndex

logger = logging.getLogger(__name__)

_MAX_REPAIR_RETRIES = 3

_CONJUGATE_METHOD_HINTS: frozenset[str] = frozenset({
    "fit", "posterior", "posterior_update", "update_hyperparameters",
    "update_params", "update_alpha", "update_beta", "update_conjugate",
})
_SUFF_STAT_HINTS: frozenset[str] = frozenset({
    "sufficient", "stat", "sum", "mean", "count", "accumulate", "aggregate",
    "n_obs", "n_samples", "successes", "failures",
})
_DISTRIBUTION_HINTS: frozenset[str] = frozenset({
    "beta", "bernoulli", "binomial", "dirichlet", "categorical", "multinomial",
    "gamma", "poisson", "normal", "gaussian", "wishart", "student",
    "distribution", "prior", "posterior", "hyperparameter", "alpha", "theta",
})
_DATA_EDGE_HINTS: frozenset[str] = frozenset({
    "data", "observation", "sample", "batch", "stats", "count",
})
_DIST_EDGE_HINTS: frozenset[str] = frozenset({
    "posterior", "distribution", "prior", "params", "hyper", "alpha", "beta",
})


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


# ---------------------------------------------------------------------------
# Conjugate heuristics
# ---------------------------------------------------------------------------


def _atom_id(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _edge_key(e: DependencyEdge) -> tuple[str, str, str, str, str, str]:
    return (
        e.source_id, e.target_id, e.output_name, e.input_name,
        e.source_type, e.target_type,
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
    text = " ".join([
        mf.name or "",
        mf.return_type or "",
        mf.docstring or "",
        mf.source_code or "",
    ])
    return _contains_hint(text, _DISTRIBUTION_HINTS)


def _method_is_fit_or_posterior(mf) -> bool:
    if _contains_hint(mf.name, _CONJUGATE_METHOD_HINTS):
        return True
    return any(_contains_hint(c, _CONJUGATE_METHOD_HINTS) for c in mf.calls)


def _choose_output(atom) -> tuple[str, str]:
    if atom.outputs:
        preferred = next(
            (o for o in atom.outputs if _contains_hint(o.name, _DATA_EDGE_HINTS | _DIST_EDGE_HINTS)),
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
            (i for i in atom.inputs if _contains_hint(i.name, _DATA_EDGE_HINTS | _DIST_EDGE_HINTS)),
            atom.inputs[0],
        )
        return preferred.name, preferred.type_desc
    return output_name, "Any"


def _select_data_atom(macro_atoms, conjugate_id: str):
    for atom in macro_atoms:
        aid = _atom_id(atom.name)
        if aid == conjugate_id:
            continue
        if _contains_hint(atom.name, frozenset({"data", "ingest", "observation", "evidence", "stats"})):
            return atom
        if any(_contains_hint(o.name, _DATA_EDGE_HINTS) for o in atom.outputs):
            return atom
    return None


def _select_distribution_atom(macro_atoms, conjugate_id: str):
    for atom in macro_atoms:
        aid = _atom_id(atom.name)
        if aid == conjugate_id:
            continue
        if atom.concept_type in {ConceptType.PRIOR_INIT, ConceptType.PRIOR_DISTRIBUTION}:
            return atom
        if _contains_hint(atom.name, frozenset({"distribution", "posterior", "construct", "build"})):
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

        should_promote = has_explicit or (has_fit_like and has_suff_stats and has_distribution)
        if should_promote and atom.concept_type != ConceptType.CONJUGATE_UPDATE:
            atom = atom.model_copy(update={"concept_type": ConceptType.CONJUGATE_UPDATE})

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

    updated_plan = plan.model_copy(update={
        "macro_atoms": updated_atoms,
        "edge_definitions": edges,
    })
    return validated.model_copy(update={"plan": updated_plan})


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


async def phase1_extract(
    state: IngesterState, config: RunnableConfig
) -> dict[str, Any]:
    """Phase 1: Deterministic AST extraction."""
    try:
        extractor = _get_extractor(state["source_path"])
        dfg = await extractor.extract_class(
            state["source_path"], state["class_name"]
        )
        return {"raw_dfg": dfg, "is_opaque": dfg.is_opaque}
    except (FileNotFoundError, ValueError) as exc:
        return {"error": str(exc), "done": True}


async def phase2_chunk(
    state: IngesterState, config: RunnableConfig
) -> dict[str, Any]:
    """Phase 2: Run the chunker sub-graph."""
    deps: IngesterDeps = config["configurable"]["deps"]

    if state.get("error"):
        return {}

    chunker_deps = ChunkerDeps(llm=deps.llm, faiss_index=deps.faiss_index)
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

    return {"validated_plan": validated}


async def phase2_conjugate_heuristics(
    state: IngesterState, config: RunnableConfig
) -> dict[str, Any]:
    """Deterministic pass: detect closed-form conjugate update logic."""
    if state.get("error"):
        return {}
    validated = state.get("validated_plan")
    dfg = state.get("raw_dfg")
    if validated is None or dfg is None:
        return {}
    updated = apply_conjugate_heuristics(dfg, validated)
    return {"validated_plan": updated}


async def phase3_emit(
    state: IngesterState, config: RunnableConfig
) -> dict[str, Any]:
    """Phase 3: Generate code and build IngestionBundle."""
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
                state["validated_plan"].plan.macro_atoms, dfg, deps.llm,
            )
            if witness_source.strip():
                bundle = bundle.model_copy(update={
                    "generated_witnesses": witness_source,
                })
        except Exception as exc:
            logger.warning("Opaque witness generation failed: %s", exc)

    return {"bundle": bundle}


async def verify_types(
    state: IngesterState, config: RunnableConfig
) -> dict[str, Any]:
    """Write generated source to temp files and run mypy."""
    deps: IngesterDeps = config["configurable"]["deps"]
    bundle: IngestionBundle = state["bundle"]

    if state.get("error"):
        return {}

    if deps.proof_env is None:
        return {"mypy_passed": True}

    # Write generated source to temp directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="ingester_"))
    atoms_path = tmp_dir / "atoms.py"
    atoms_path.write_text(bundle.generated_atoms)

    if bundle.generated_state_models:
        models_path = tmp_dir / "state_models.py"
        models_path.write_text(bundle.generated_state_models)

    try:
        ok, output = await deps.proof_env.check_proof(
            bundle.generated_atoms, ""
        )
        if ok:
            return {"mypy_passed": True, "mypy_errors": ""}
        else:
            return {"mypy_passed": False, "mypy_errors": output}
    except Exception as exc:
        logger.warning("mypy verification failed: %s", exc)
        return {"mypy_passed": False, "mypy_errors": str(exc)}


async def verify_ghost(
    state: IngesterState, config: RunnableConfig
) -> dict[str, Any]:
    """Run ghost simulation on the generated CDG."""
    bundle: IngestionBundle = state["bundle"]

    if state.get("error"):
        return {}

    try:
        from ageom.synthesizer.ghost_sim import run_ghost_simulation

        report = run_ghost_simulation(bundle.cdg, bundle.match_results)
        bundle_update = bundle.model_copy(update={
            "ghost_sim_passed": report.passed or not report.ran,
            "ghost_sim_report": {
                "ran": report.ran,
                "passed": report.passed,
                "error": report.error,
                "error_node": report.error_node,
                "error_function": report.error_function,
                "coverage": report.coverage,
            },
        })
        passed = report.passed or not report.ran
        errors = report.error if not passed else ""
        return {
            "bundle": bundle_update,
            "ghost_passed": passed,
            "ghost_errors": errors,
        }
    except ImportError:
        return {"ghost_passed": True, "ghost_errors": ""}


async def repair_types(
    state: IngesterState, config: RunnableConfig
) -> dict[str, Any]:
    """LLM-assisted repair of mypy type errors."""
    deps: IngesterDeps = config["configurable"]["deps"]
    bundle: IngestionBundle = state["bundle"]

    user_prompt = FIX_TYPE_ERROR_USER.format(
        mypy_errors=state.get("mypy_errors", ""),
        source_code=bundle.generated_atoms,
    )

    from ageom.llm_router import INGESTER_FIX_TYPE, select_llm

    response = await select_llm(deps.llm, INGESTER_FIX_TYPE).complete(FIX_TYPE_ERROR_SYSTEM, user_prompt)

    try:
        fixes = json.loads(response)
        if isinstance(fixes, list) and fixes:
            lines = bundle.generated_atoms.splitlines()
            for fix in fixes:
                start = fix.get("line_start", 1) - 1
                end = fix.get("line_end", start + 1)
                replacement = fix.get("replacement", "")
                lines[start:end] = replacement.splitlines()
            updated_atoms = "\n".join(lines)
            updated_bundle = bundle.model_copy(update={
                "generated_atoms": updated_atoms,
            })
            return {
                "bundle": updated_bundle,
                "type_repair_count": state.get("type_repair_count", 0) + 1,
            }
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Failed to parse type repair response: %s", exc)

    return {"type_repair_count": state.get("type_repair_count", 0) + 1}


async def repair_ghost(
    state: IngesterState, config: RunnableConfig
) -> dict[str, Any]:
    """LLM-assisted repair of ghost simulation errors."""
    deps: IngesterDeps = config["configurable"]["deps"]
    bundle: IngestionBundle = state["bundle"]
    report = bundle.ghost_sim_report

    user_prompt = FIX_GHOST_ERROR_USER.format(
        error_node=report.get("error_node", ""),
        error_function=report.get("error_function", ""),
        error_message=report.get("error", ""),
        witness_source=bundle.generated_witnesses,
    )

    from ageom.llm_router import INGESTER_FIX_GHOST, select_llm

    response = await select_llm(deps.llm, INGESTER_FIX_GHOST).complete(FIX_GHOST_ERROR_SYSTEM, user_prompt)

    try:
        fixes = json.loads(response)
        if isinstance(fixes, list) and fixes:
            updated_witnesses = bundle.generated_witnesses
            for fix in fixes:
                replacement = fix.get("replacement", "")
                if replacement:
                    updated_witnesses = replacement
                    break
            updated_bundle = bundle.model_copy(update={
                "generated_witnesses": updated_witnesses,
            })
            return {
                "bundle": updated_bundle,
                "ghost_repair_count": state.get("ghost_repair_count", 0) + 1,
            }
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Failed to parse ghost repair response: %s", exc)

    return {"ghost_repair_count": state.get("ghost_repair_count", 0) + 1}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_type_check(state: IngesterState) -> str:
    if state.get("mypy_passed", False):
        return "verify_ghost"
    if state.get("type_repair_count", 0) >= _MAX_REPAIR_RETRIES:
        return "end"
    return "repair_types"


def route_after_ghost(state: IngesterState) -> str:
    if state.get("ghost_passed", False):
        return "end"
    if state.get("ghost_repair_count", 0) >= _MAX_REPAIR_RETRIES:
        return "end"
    return "repair_ghost"


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
        },
    )
    graph.add_edge("repair_ghost", "verify_types")

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
    ) -> None:
        global _MAX_REPAIR_RETRIES
        _MAX_REPAIR_RETRIES = max_repair_retries

        self._deps = IngesterDeps(
            llm=llm,
            proof_env=proof_env,
            faiss_index=faiss_index,
            output_dir=output_dir,
        )
        self._graph = build_ingester_graph().compile()

    async def ingest(
        self, source_path: str, class_name: str
    ) -> IngestionBundle:
        """Run the full ingester pipeline.

        Args:
            source_path: Path to the Python source file.
            class_name: Name of the class to ingest.

        Returns:
            An ``IngestionBundle`` with CDG, generated source, and match results.
        """
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
            "type_repair_count": 0,
            "ghost_repair_count": 0,
            "done": False,
            "error": "",
        }

        config: dict[str, Any] = {
            "configurable": {"deps": self._deps}
        }

        final_state = await self._graph.ainvoke(initial_state, config=config)

        bundle: IngestionBundle = final_state["bundle"]
        return bundle

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
        extractor = _get_extractor(source_path)
        dfg = await extractor.extract_procedural(source_path, name)
        plan = build_procedural_plan(dfg, name)
        return emit_ingestion_bundle(
            plan, name, source_path,
            source_language=dfg.source_language,
        )
