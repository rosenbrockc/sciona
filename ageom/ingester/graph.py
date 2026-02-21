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

    response = await deps.llm.complete(FIX_TYPE_ERROR_SYSTEM, user_prompt)

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

    response = await deps.llm.complete(FIX_GHOST_ERROR_SYSTEM, user_prompt)

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
    graph.add_node("phase3_emit", phase3_emit)
    graph.add_node("verify_types", verify_types)
    graph.add_node("verify_ghost", verify_ghost)
    graph.add_node("repair_types", repair_types)
    graph.add_node("repair_ghost", repair_ghost)

    graph.set_entry_point("phase1_extract")
    graph.add_edge("phase1_extract", "phase2_chunk")
    graph.add_edge("phase2_chunk", "phase3_emit")
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
