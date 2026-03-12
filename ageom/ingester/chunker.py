"""Phase 2: LangGraph semantic chunking sub-graph.

Groups methods into macro-atoms, hoists cross-window state, searches
for existing sub-atoms, and validates coverage via a critic loop.
"""

from __future__ import annotations

import asyncio
import json
import logging

from ageom.json_utils import extract_json
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from ageom.hunter.llm import LLMClient
from ageom.architect.models import ConceptType, IOSpec
from ageom.ingester.monitor import IngestMonitor
from ageom.ingester.models import (
    ConceptualProfile,
    DependencyEdge,
    MacroAtomSpec,
    ProposedMacroPlan,
    RawDataFlowGraph,
    StateModelSpec,
    SubAtomRef,
    ValidatedMacroPlan,
)
from ageom.ingester.prompts import (
    CONCEPTUAL_ABSTRACT_SYSTEM,
    CONCEPTUAL_ABSTRACT_USER,
    DECOMPOSE_ATOM_SYSTEM,
    DECOMPOSE_ATOM_USER,
    HOIST_STATE_SYSTEM,
    HOIST_STATE_USER,
    SEMANTIC_CHUNK_SYSTEM,
    SEMANTIC_CHUNK_USER,
)
from ageom.llm_router import (
    INGESTER_ABSTRACT,
    INGESTER_CHUNK,
    INGESTER_DECOMPOSE,
    INGESTER_HOIST_STATE,
    select_llm,
)
from ageom.shared_context import (
    SharedContextMetrics,
    SharedContextStore,
    format_context_block,
)
from ageom.protocols import SemanticIndex

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# State & deps
# ---------------------------------------------------------------------------


class ChunkerState(TypedDict):
    raw_dfg: RawDataFlowGraph
    proposed_plan: ProposedMacroPlan
    validated_plan: ValidatedMacroPlan
    critique_passed: bool
    critique_reason: str
    retry_count: int
    missing_attrs: list[str]
    done: bool


@dataclass
class ChunkerDeps:
    llm: LLMClient
    faiss_index: SemanticIndex | None = None
    max_depth: int = 1
    line_threshold: int = 30
    monitor: IngestMonitor | None = None
    shared_context: SharedContextStore | None = None
    shared_context_metrics: SharedContextMetrics | None = None
    context_namespace: str = ""
    context_budget_chars: int = 900
    parallelism: int = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_monitor(config: RunnableConfig) -> IngestMonitor | None:
    deps = config.get("configurable", {}).get("deps")
    if deps is None:
        return None
    return getattr(deps, "monitor", None)


def _build_method_summaries(dfg: RawDataFlowGraph) -> str:
    lines = []
    for mf in dfg.methods:
        reads = ", ".join(mf.reads) if mf.reads else "(none)"
        writes = ", ".join(mf.writes) if mf.writes else "(none)"
        calls = ", ".join(mf.calls) if mf.calls else "(none)"
        lines.append(
            f"- {mf.name}({', '.join(mf.params)})"
            f"\n  reads: {reads}"
            f"\n  writes: {writes}"
            f"\n  calls: {calls}"
        )
    return "\n".join(lines)


def _build_attr_graph(dfg: RawDataFlowGraph) -> str:
    lines = []
    for attr, accesses in sorted(dfg.all_attributes.items()):
        lines.append(f"  {attr}: {accesses}")
    return "\n".join(lines)


def _build_config_branches(dfg: RawDataFlowGraph) -> str:
    if not dfg.config_branches:
        return "(none)"
    lines = []
    for cb in dfg.config_branches:
        lines.append(
            f"  if self.options.{cb.config_attr} in {cb.method} "
            f"(lines {cb.lines[0]}-{cb.lines[1]})"
        )
    return "\n".join(lines)


async def _search_context(
    deps: ChunkerDeps,
    *,
    channel: str,
    query: str,
    limit: int = 3,
) -> str:
    store = deps.shared_context
    ns = deps.context_namespace
    if store is None or not ns:
        return ""
    try:
        records = await store.search(f"{ns}/{channel}", query, limit=limit)
        return format_context_block(
            "Shared Context",
            records,
            max_chars=deps.context_budget_chars,
            metrics=deps.shared_context_metrics,
        )
    except Exception:
        return ""


async def _put_context(
    deps: ChunkerDeps,
    *,
    channel: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    store = deps.shared_context
    ns = deps.context_namespace
    if store is None or not ns:
        return
    try:
        await store.put(f"{ns}/{channel}", text, metadata=metadata)
    except Exception:
        return


def _compute_state_edges(
    dfg: RawDataFlowGraph,
    plan: ProposedMacroPlan,
) -> list[DependencyEdge]:
    """Compute deterministic state-typed edges from method read/write sets.

    For each state attribute, creates edges from every writer atom to every
    reader atom (deduplicated, self-loops excluded).  The edge type is the
    state model name (e.g. ``RollingAveragerState``).
    """
    if not plan.state_models:
        return []

    state_model = plan.state_models[0]
    state_model_name = state_model.model_name
    state_attrs = set(state_model.source_attrs)

    # Map method name -> atom snake_case id
    method_to_atom: dict[str, str] = {}
    for atom in plan.macro_atoms:
        atom_id = atom.name.lower().replace(" ", "_").replace("-", "_")
        for mname in atom.method_names:
            method_to_atom[mname] = atom_id

    # For each method, check if it reads/writes state attrs
    writers: dict[str, set[str]] = {}  # attr -> set of atom_ids
    readers: dict[str, set[str]] = {}  # attr -> set of atom_ids

    for mf in dfg.methods:
        if mf.name == "__init__":
            continue
        atom_id = method_to_atom.get(mf.name)
        if atom_id is None:
            continue
        for attr in mf.writes:
            if attr in state_attrs:
                writers.setdefault(attr, set()).add(atom_id)
        for attr in mf.reads:
            if attr in state_attrs:
                readers.setdefault(attr, set()).add(atom_id)

    # Build edges: writer -> reader for each attr
    seen: set[tuple[str, str]] = set()
    edges: list[DependencyEdge] = []

    for attr in state_attrs:
        for w in writers.get(attr, set()):
            for r in readers.get(attr, set()):
                if w == r:
                    continue
                key = (w, r)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    DependencyEdge(
                        source_id=w,
                        target_id=r,
                        output_name=attr,
                        input_name=attr,
                        source_type=state_model_name,
                        target_type=state_model_name,
                    )
                )

    return edges


def _parse_macro_atoms(raw: dict) -> list[MacroAtomSpec]:
    atoms = []
    for item in raw.get("macro_atoms", []):
        inputs = [
            IOSpec(
                name=io.get("name", ""),
                type_desc=io.get("type_desc", ""),
                constraints=io.get("constraints", ""),
            )
            for io in item.get("inputs", [])
        ]
        outputs = [
            IOSpec(
                name=io.get("name", ""),
                type_desc=io.get("type_desc", ""),
                constraints=io.get("constraints", ""),
            )
            for io in item.get("outputs", [])
        ]
        try:
            concept = ConceptType(item.get("concept_type", "custom"))
        except ValueError:
            logger.warning(
                "Unknown concept_type %r for node %r, falling back to custom",
                item.get("concept_type"),
                item.get("name"),
            )
            concept = ConceptType.CUSTOM
        atoms.append(
            MacroAtomSpec(
                name=item.get("name", ""),
                description=item.get("description", ""),
                method_names=item.get("method_names", []),
                inputs=inputs,
                outputs=outputs,
                config_params=item.get("config_params", []),
                concept_type=concept,
                is_optional=item.get("is_optional", False),
                is_stochastic=item.get("is_stochastic", False),
                requires_rng_key=item.get("requires_rng_key", False),
                requires_autodiff=item.get("requires_autodiff", False),
                autodiff_backend=item.get("autodiff_backend", ""),
            )
        )
    return atoms


def _parse_edges(raw: dict) -> list[DependencyEdge]:
    edges = []
    for item in raw.get("edges", []):
        edges.append(
            DependencyEdge(
                source_id=item.get("source_id", ""),
                target_id=item.get("target_id", ""),
                output_name=item.get("output_name", ""),
                input_name=item.get("input_name", ""),
                source_type=item.get("source_type", ""),
                target_type=item.get("target_type", ""),
            )
        )
    return edges


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


async def propose_macro_atoms(
    state: ChunkerState, config: RunnableConfig
) -> dict[str, Any]:
    """LLM call: group methods into macro-atoms."""
    deps: ChunkerDeps = config["configurable"]["deps"]
    mon = _get_monitor(config)
    if mon:
        mon.heartbeat(phase="phase2_chunk", step="propose_macro_atoms")
    dfg = state["raw_dfg"]

    # Opaque DL boundary: deterministic single-atom plan, no LLM cost
    if dfg.is_opaque and dfg.methods:
        mf = dfg.methods[0]
        atom = MacroAtomSpec(
            name=dfg.class_name,
            description=mf.docstring or f"Opaque DL boundary: {dfg.class_name}",
            method_names=[mf.name],
            inputs=[IOSpec(name=p, type_desc="Any") for p in mf.params],
            outputs=[IOSpec(name="output", type_desc=mf.return_type or "Any")],
            concept_type=ConceptType.NEURAL_NETWORK,
            is_opaque=True,
        )
        return {"proposed_plan": ProposedMacroPlan(macro_atoms=[atom])}

    retry_context = ""
    if state.get("retry_count", 0) > 0:
        missing = state.get("missing_attrs", [])
        retry_context = (
            f"RETRY {state['retry_count']}/{_MAX_RETRIES}. "
            f"Previous attempt missed these attributes: {missing}. "
            f"Every self.* attribute MUST appear in at least one macro-atom."
        )

    user_prompt = SEMANTIC_CHUNK_USER.format(
        class_name=dfg.class_name,
        method_summaries=_build_method_summaries(dfg),
        attr_graph=_build_attr_graph(dfg),
        config_branches=_build_config_branches(dfg),
        retry_context=retry_context,
    )

    if mon:
        mon.llm_start(INGESTER_CHUNK)
    try:
        response = await select_llm(deps.llm, INGESTER_CHUNK).complete(
            SEMANTIC_CHUNK_SYSTEM, user_prompt
        )
        if mon:
            mon.llm_end(INGESTER_CHUNK, ok=True)
    except Exception as exc:
        if mon:
            mon.llm_end(INGESTER_CHUNK, ok=False, error=str(exc))
        raise

    try:
        raw = extract_json(response)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON, using empty plan")
        raw = {"macro_atoms": [], "edges": []}

    macro_atoms = _parse_macro_atoms(raw)
    edges = _parse_edges(raw)

    plan = ProposedMacroPlan(
        macro_atoms=macro_atoms,
        edge_definitions=edges,
    )
    return {"proposed_plan": plan}


async def flatten_config(state: ChunkerState, config: RunnableConfig) -> dict[str, Any]:
    """Deterministic: Flatten config-gated branches into optional variants."""
    mon = _get_monitor(config)
    if mon:
        mon.heartbeat(phase="phase2_chunk", step="flatten_config")
    plan = state["proposed_plan"]
    dfg = state["raw_dfg"]

    new_atoms = []
    for atom in plan.macro_atoms:
        branches = []
        for mname in atom.method_names:
            mf = next((m for m in dfg.methods if m.name == mname), None)
            if mf:
                branches.extend(mf.config_branches)

        if not branches:
            new_atoms.append(atom)
            continue

        # Analyze branches to see if we should split
        # For now, we simply ensure the atom is marked is_optional if it has significant branching
        # and append it. In a full implementation, we would duplicate the atom for each variant.
        # Here we simulate the flattening by marking it for the orchestrator.
        if len(branches) > 0 and not atom.is_optional:
            # It has config branches but wasn"t marked optional.
            # We keep it as is but could tag it in description.
            atom.description += f" (Contains {len(branches)} config branches)"
            # atom.is_optional = True # Optional implies it might not run. Branching implies one of many runs.

        new_atoms.append(atom)

    updated = plan.model_copy(update={"macro_atoms": new_atoms})
    return {"proposed_plan": updated}


async def hoist_state(state: ChunkerState, config: RunnableConfig) -> dict[str, Any]:
    """LLM call: identify cross-window attrs and generate state model specs."""
    deps: ChunkerDeps = config["configurable"]["deps"]
    mon = _get_monitor(config)
    if mon:
        mon.heartbeat(phase="phase2_chunk", step="hoist_state")
    dfg = state["raw_dfg"]
    plan = state["proposed_plan"]

    if not dfg.cross_window_attrs:
        return {"proposed_plan": plan}

    macro_plan_json = json.dumps([a.model_dump() for a in plan.macro_atoms], indent=2)
    user_prompt = HOIST_STATE_USER.format(
        cross_window_attrs=dfg.cross_window_attrs,
        macro_plan_json=macro_plan_json,
    )

    if mon:
        mon.llm_start(INGESTER_HOIST_STATE)
    try:
        response = await select_llm(deps.llm, INGESTER_HOIST_STATE).complete(
            HOIST_STATE_SYSTEM, user_prompt
        )
        if mon:
            mon.llm_end(INGESTER_HOIST_STATE, ok=True)
    except Exception as exc:
        if mon:
            mon.llm_end(INGESTER_HOIST_STATE, ok=False, error=str(exc))
        raise

    try:
        raw = extract_json(response)
    except json.JSONDecodeError:
        logger.warning("Failed to parse state hoisting response")
        return {"proposed_plan": plan}

    state_models = []
    for item in raw.get("state_models", []):
        fields = [tuple(f) for f in item.get("fields", [])]
        state_models.append(
            StateModelSpec(
                model_name=item.get("model_name", ""),
                fields=fields,
                source_attrs=item.get("source_attrs", []),
                docstring=item.get("docstring", ""),
            )
        )

    updated = plan.model_copy(update={"state_models": state_models})

    # Compute deterministic state edges and append to existing edges
    state_edges = _compute_state_edges(dfg, updated)
    if state_edges:
        all_edges = list(updated.edge_definitions) + state_edges
        updated = updated.model_copy(update={"edge_definitions": all_edges})

    return {"proposed_plan": updated}


async def search_sub_atoms(
    state: ChunkerState, config: RunnableConfig
) -> dict[str, Any]:
    """Deterministic: query FAISS index for existing atoms matching operations."""
    deps: ChunkerDeps = config["configurable"]["deps"]
    mon = _get_monitor(config)
    if mon:
        mon.heartbeat(phase="phase2_chunk", step="search_sub_atoms")
    plan = state["proposed_plan"]

    if deps.faiss_index is None:
        return {"proposed_plan": plan}

    sub_refs: list[SubAtomRef] = []
    for atom in plan.macro_atoms:
        results = deps.faiss_index.search_by_embedding(atom.name, k=3)
        for decl, score in results:
            if score > 0.5:
                sub_refs.append(
                    SubAtomRef(
                        atom_name=decl.name,
                        similarity_score=score,
                    )
                )

    updated = plan.model_copy(update={"sub_atom_refs": sub_refs})
    return {"proposed_plan": updated}


async def critic_validate(
    state: ChunkerState, config: RunnableConfig
) -> dict[str, Any]:
    """Validate that ALL self.* attributes appear in macro-atoms or state models."""
    mon = _get_monitor(config)
    if mon:
        mon.heartbeat(phase="phase2_chunk", step="critic_validate")
    dfg = state["raw_dfg"]
    plan = state["proposed_plan"]

    # Opaque DL boundary: auto-pass (no self.* tracking)
    if dfg.is_opaque:
        return {
            "validated_plan": ValidatedMacroPlan(
                plan=plan,
                all_attrs_accounted=True,
                coverage_report="Opaque DL boundary: no self.* tracking required.",
            ),
            "critique_passed": True,
            "critique_reason": "",
        }

    # Collect all attrs covered by macro-atoms
    covered_attrs: set[str] = set()
    for atom in plan.macro_atoms:
        for mname in atom.method_names:
            mf = next((m for m in dfg.methods if m.name == mname), None)
            if mf:
                covered_attrs.update(mf.reads)
                covered_attrs.update(mf.writes)

    # Attrs covered by state models
    for sm in plan.state_models:
        covered_attrs.update(sm.source_attrs)

    # Check coverage
    all_attrs = set(dfg.all_attributes.keys())
    missing = all_attrs - covered_attrs

    if not missing:
        validated = ValidatedMacroPlan(
            plan=plan,
            all_attrs_accounted=True,
            coverage_report="All attributes accounted for.",
        )
        return {
            "validated_plan": validated,
            "critique_passed": True,
            "critique_reason": "",
        }
    else:
        missing_list = sorted(missing)
        return {
            "critique_passed": False,
            "critique_reason": f"Missing attributes: {missing_list}",
            "missing_attrs": missing_list,
        }


async def prepare_chunk_retry(
    state: ChunkerState, config: RunnableConfig
) -> dict[str, Any]:
    """Increment retry counter for the next proposal attempt."""
    mon = _get_monitor(config)
    if mon:
        mon.heartbeat(phase="phase2_chunk", step="prepare_chunk_retry")
    return {"retry_count": state.get("retry_count", 0) + 1}


# ---------------------------------------------------------------------------
# Complexity heuristic + recursive decomposition
# ---------------------------------------------------------------------------


def is_atom_complex(
    atom: MacroAtomSpec,
    dfg: RawDataFlowGraph,
    line_threshold: int = 30,
) -> bool:
    """Determine whether *atom* is too complex and should be decomposed.

    An atom is complex when ANY of:
    - Combined method source exceeds *line_threshold* lines
    - Methods call 3+ internal sub-functions
    - Any method body is ``NotImplementedError`` (skeleton stub)
    """
    total_lines = 0
    total_internal_calls = 0
    has_not_implemented = False

    by_name = {m.name: m for m in dfg.methods}
    for mname in atom.method_names:
        mf = by_name.get(mname)
        if mf is None:
            continue
        src = mf.source_code or ""
        total_lines += len(src.strip().splitlines())
        # Count calls to other methods in the same class
        internal_methods = {m.name for m in dfg.methods}
        total_internal_calls += sum(1 for c in mf.calls if c in internal_methods)
        if "NotImplementedError" in src:
            has_not_implemented = True

    # Also honour the pre-populated source_lines field
    if atom.source_lines > 0:
        total_lines = max(total_lines, atom.source_lines)

    return total_lines > line_threshold or total_internal_calls >= 3 or has_not_implemented


def _gather_source_for_atom(
    atom: MacroAtomSpec,
    dfg: RawDataFlowGraph,
) -> tuple[str, list[str]]:
    """Return (combined_source, internal_calls) for an atom's methods."""
    by_name = {m.name: m for m in dfg.methods}
    sources: list[str] = []
    all_calls: list[str] = []
    for mname in atom.method_names:
        mf = by_name.get(mname)
        if mf and mf.source_code:
            sources.append(mf.source_code)
            all_calls.extend(mf.calls)
    return "\n\n".join(sources), all_calls


def _parse_sub_atoms(raw: dict, parent_depth: int) -> tuple[list[MacroAtomSpec], list[DependencyEdge]]:
    """Parse LLM decomposition response into sub-atoms and edges."""
    sub_atoms: list[MacroAtomSpec] = []
    for item in raw.get("sub_atoms", []):
        inputs = [
            IOSpec(
                name=io.get("name", ""),
                type_desc=io.get("type_desc", ""),
                constraints=io.get("constraints", ""),
            )
            for io in item.get("inputs", [])
        ]
        outputs = [
            IOSpec(
                name=io.get("name", ""),
                type_desc=io.get("type_desc", ""),
                constraints=io.get("constraints", ""),
            )
            for io in item.get("outputs", [])
        ]
        try:
            concept = ConceptType(item.get("concept_type", "custom"))
        except ValueError:
            logger.warning(
                "Unknown concept_type %r for node %r, falling back to custom",
                item.get("concept_type"),
                item.get("name"),
            )
            concept = ConceptType.CUSTOM
        sub_atoms.append(
            MacroAtomSpec(
                name=item.get("name", ""),
                description=item.get("description", ""),
                inputs=inputs,
                outputs=outputs,
                concept_type=concept,
                depth=parent_depth + 1,
            )
        )

    edges: list[DependencyEdge] = []
    for item in raw.get("edges", []):
        edges.append(
            DependencyEdge(
                source_id=_snake_case_id(item.get("source_id", "")),
                target_id=_snake_case_id(item.get("target_id", "")),
                output_name=item.get("output_name", ""),
                input_name=item.get("input_name", ""),
                source_type=item.get("source_type", ""),
                target_type=item.get("target_type", ""),
            )
        )
    return sub_atoms, edges


def _snake_case_id(name: str) -> str:
    """Normalize an edge endpoint to snake_case (matching _emit_atom_nodes)."""
    return name.lower().replace(" ", "_").replace("-", "_")


async def _decompose_single_atom(
    atom: MacroAtomSpec,
    dfg: RawDataFlowGraph,
    llm: LLMClient,
    max_depth: int,
    line_threshold: int,
    current_depth: int,
    monitor: IngestMonitor | None = None,
    shared_context: SharedContextStore | None = None,
    shared_context_metrics: SharedContextMetrics | None = None,
    context_namespace: str = "",
    context_budget_chars: int = 900,
) -> MacroAtomSpec:
    """Recursively decompose a single atom if it exceeds complexity thresholds."""
    if current_depth >= max_depth:
        return atom

    if not is_atom_complex(atom, dfg, line_threshold):
        return atom

    source_code, internal_calls = _gather_source_for_atom(atom, dfg)
    if not source_code.strip():
        return atom

    # Try deterministic control-flow decomposition first
    from ageom.ingester.control_flow_decomposer import decompose_function

    primary_method = atom.method_names[0] if atom.method_names else atom.name
    cf_result = decompose_function(source_code, primary_method)
    if cf_result is not None and cf_result.confidence >= 0.5:
        children = []
        for sub in cf_result.sub_atoms:
            child = MacroAtomSpec(
                name=sub.name,
                description=sub.description,
                method_names=[],
                inputs=[IOSpec(name=i, type_desc="") for i in sub.inputs],
                outputs=[IOSpec(name=o, type_desc="") for o in sub.outputs],
                concept_type=atom.concept_type,
                depth=current_depth + 1,
                source_lines=list(range(sub.source_lines[0], sub.source_lines[1] + 1)),
            )
            children.append(child)
        if children:
            atom.children = children
            atom.sub_edges = [
                (e["from"], e["to"], e.get("data", ""))
                for e in cf_result.edges
            ]
            logger.info(
                "Deterministic CFG decomposition for %s: %d sub-atoms (confidence=%.2f)",
                atom.name,
                len(children),
                cf_result.confidence,
            )
            return atom

    inputs_str = ", ".join(f"{i.name}: {i.type_desc}" for i in atom.inputs) or "(none)"
    outputs_str = ", ".join(f"{o.name}: {o.type_desc}" for o in atom.outputs) or "(none)"
    calls_str = ", ".join(internal_calls) if internal_calls else "(none)"

    user_prompt = DECOMPOSE_ATOM_USER.format(
        atom_name=atom.name,
        atom_description=atom.description,
        current_inputs=inputs_str,
        current_outputs=outputs_str,
        internal_calls=calls_str,
        source_code=source_code,
    )
    if shared_context is not None and context_namespace:
        try:
            records = await shared_context.search(
                f"{context_namespace}/decompose",
                f"{atom.name} {atom.description}",
                limit=3,
            )
            block = format_context_block(
                "Shared Context",
                records,
                max_chars=context_budget_chars,
                metrics=shared_context_metrics,
            )
            if block:
                user_prompt += f"\n\n{block}"
        except Exception:
            pass

    try:
        if monitor:
            monitor.llm_start(INGESTER_DECOMPOSE)
        response = await select_llm(llm, INGESTER_DECOMPOSE).complete(
            DECOMPOSE_ATOM_SYSTEM, user_prompt
        )
    except Exception as exc:
        if monitor:
            monitor.llm_end(INGESTER_DECOMPOSE, ok=False, error=str(exc))
        logger.warning("Decomposition failed for %s: %s", atom.name, exc)
        return atom

    if monitor:
        monitor.llm_end(INGESTER_DECOMPOSE, ok=True)
    try:
        raw = extract_json(response)
    except Exception as exc:
        logger.warning("Decomposition JSON parse failed for %s: %s", atom.name, exc)
        return atom

    children, sub_edges = _parse_sub_atoms(raw, current_depth)
    if not children:
        return atom
    if shared_context is not None and context_namespace:
        try:
            await shared_context.put(
                f"{context_namespace}/decompose",
                (
                    f"Atom: {atom.name}\n"
                    f"Children: {', '.join(child.name for child in children)}\n"
                    f"Sub-edge count: {len(sub_edges)}"
                ),
                metadata={"atom_name": atom.name, "depth": current_depth},
            )
        except Exception:
            pass

    # Recurse into children
    recursed_children: list[MacroAtomSpec] = []
    for child in children:
        recursed = await _decompose_single_atom(
            child,
            dfg,
            llm,
            max_depth,
            line_threshold,
            current_depth + 1,
            monitor=monitor,
            shared_context=shared_context,
            shared_context_metrics=shared_context_metrics,
            context_namespace=context_namespace,
            context_budget_chars=context_budget_chars,
        )
        recursed_children.append(recursed)

    return atom.model_copy(update={"children": recursed_children, "sub_edges": sub_edges})


async def decompose_complex_atoms(
    state: ChunkerState, config: RunnableConfig
) -> dict[str, Any]:
    """Recursively decompose complex atoms based on complexity heuristics."""
    deps: ChunkerDeps = config["configurable"]["deps"]
    mon = _get_monitor(config)
    if mon:
        mon.heartbeat(phase="phase2_chunk", step="decompose_complex_atoms")
    validated = state["validated_plan"]
    plan = validated.plan
    dfg = state["raw_dfg"]

    max_depth = deps.max_depth
    line_threshold = deps.line_threshold

    # At depth 1 (default), skip decomposition entirely for backward compat
    if max_depth <= 1:
        return {"validated_plan": validated}

    parallelism = max(1, deps.parallelism)
    if parallelism <= 1 or len(plan.macro_atoms) <= 1:
        decomposed_atoms: list[MacroAtomSpec] = []
        for atom in plan.macro_atoms:
            result = await _decompose_single_atom(
                atom,
                dfg,
                deps.llm,
                max_depth,
                line_threshold,
                current_depth=1,
                monitor=mon,
                shared_context=deps.shared_context,
                shared_context_metrics=deps.shared_context_metrics,
                context_namespace=deps.context_namespace,
                context_budget_chars=deps.context_budget_chars,
            )
            decomposed_atoms.append(result)
    else:
        semaphore = asyncio.Semaphore(parallelism)

        async def _run(atom: MacroAtomSpec) -> MacroAtomSpec:
            async with semaphore:
                return await _decompose_single_atom(
                    atom,
                    dfg,
                    deps.llm,
                    max_depth,
                    line_threshold,
                    current_depth=1,
                    monitor=mon,
                    shared_context=deps.shared_context,
                    shared_context_metrics=deps.shared_context_metrics,
                    context_namespace=deps.context_namespace,
                    context_budget_chars=deps.context_budget_chars,
                )

        decomposed_atoms = list(await asyncio.gather(*[_run(a) for a in plan.macro_atoms]))

    new_plan = plan.model_copy(update={"macro_atoms": decomposed_atoms})
    new_validated = validated.model_copy(update={"plan": new_plan})
    return {"validated_plan": new_validated}


def _format_io_specs(specs: list) -> str:
    """Format IOSpec list for the abstraction prompt."""
    if not specs:
        return "(none)"
    return "\n".join(
        f"  - {s.name}: {s.type_desc}"
        + (f" ({s.constraints})" if s.constraints else "")
        for s in specs
    )


def _build_enriched_description(original: str, profile: ConceptualProfile) -> str:
    """Merge a ConceptualProfile into an atom description.

    The profile JSON is appended as a fenced block so downstream consumers
    (the Hunter's FAISS index, CDG node descriptions, generated docstrings)
    all benefit from the domain-agnostic vocabulary.
    """
    profile_json = json.dumps(profile.model_dump(), indent=2)
    return f"{original}\n\n<!-- conceptual_profile -->\n{profile_json}\n<!-- /conceptual_profile -->"


def _parse_conceptual_profile(raw: dict) -> ConceptualProfile:
    """Parse a raw LLM JSON response into a ConceptualProfile."""
    return ConceptualProfile(
        abstract_name=raw.get("abstract_name", ""),
        conceptual_transform=raw.get("conceptual_transform", ""),
        abstract_inputs=raw.get("abstract_inputs", []),
        abstract_outputs=raw.get("abstract_outputs", []),
        algorithmic_properties=raw.get("algorithmic_properties", []),
        cross_disciplinary_applications=raw.get("cross_disciplinary_applications", []),
    )


async def abstract_atoms(state: ChunkerState, config: RunnableConfig) -> dict[str, Any]:
    """LLM call: generate domain-agnostic conceptual profiles for each atom.

    Runs the Conceptual Abstraction Agent on every atom in the validated
    plan.  The resulting profile is stored on each atom's
    ``conceptual_profile`` field.  A plain-text summary is threaded
    through to the FAISS index via the emitter (not embedded in the
    description).
    """
    deps: ChunkerDeps = config["configurable"]["deps"]
    mon = _get_monitor(config)
    if mon:
        mon.heartbeat(phase="phase2_chunk", step="abstract_atoms")
    validated = state["validated_plan"]
    plan = validated.plan

    async def _abstract_one(atom: MacroAtomSpec) -> MacroAtomSpec:
        user_prompt = CONCEPTUAL_ABSTRACT_USER.format(
            atom_name=atom.name,
            atom_description=atom.description,
            concept_type=atom.concept_type.value,
            inputs_spec=_format_io_specs(atom.inputs),
            outputs_spec=_format_io_specs(atom.outputs),
            method_names=", ".join(atom.method_names),
        )
        shared_block = await _search_context(
            deps,
            channel="abstract",
            query=f"{atom.name} {atom.concept_type.value}",
            limit=3,
        )
        if shared_block:
            user_prompt += f"\n\n{shared_block}"

        try:
            if mon:
                mon.llm_start(INGESTER_ABSTRACT)
            response = await select_llm(deps.llm, INGESTER_ABSTRACT).complete(
                CONCEPTUAL_ABSTRACT_SYSTEM, user_prompt
            )
            if mon:
                mon.llm_end(INGESTER_ABSTRACT, ok=True)
        except Exception as exc:
            if mon:
                mon.llm_end(INGESTER_ABSTRACT, ok=False, error=str(exc))
            logger.warning("Conceptual abstraction failed for %s: %s", atom.name, exc)
            profile = ConceptualProfile(abstract_name=atom.name)
            return atom.model_copy(update={"conceptual_profile": profile})

        try:
            raw = extract_json(response)
            profile = _parse_conceptual_profile(raw)
        except Exception as exc:
            logger.warning(
                "Conceptual abstraction JSON parse failed for %s: %s", atom.name, exc
            )
            profile = ConceptualProfile(abstract_name=atom.name)

        await _put_context(
            deps,
            channel="abstract",
            text=(
                f"Atom: {atom.name}\n"
                f"Abstract name: {profile.abstract_name}\n"
                f"Transform: {profile.conceptual_transform}"
            ),
            metadata={"atom_name": atom.name, "concept_type": atom.concept_type.value},
        )
        return atom.model_copy(
            update={
                "conceptual_profile": profile,
            }
        )

    parallelism = max(1, deps.parallelism)
    if parallelism <= 1 or len(plan.macro_atoms) <= 1:
        enriched_atoms = [await _abstract_one(atom) for atom in plan.macro_atoms]
    else:
        semaphore = asyncio.Semaphore(parallelism)

        async def _run(atom: MacroAtomSpec) -> MacroAtomSpec:
            async with semaphore:
                return await _abstract_one(atom)

        enriched_atoms = list(await asyncio.gather(*[_run(a) for a in plan.macro_atoms]))

    new_plan = plan.model_copy(update={"macro_atoms": enriched_atoms})
    new_validated = validated.model_copy(update={"plan": new_plan})
    return {"validated_plan": new_validated}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_critic(state: ChunkerState) -> str:
    if state.get("critique_passed", False):
        return "end"
    if state.get("retry_count", 0) >= _MAX_RETRIES:
        return "end_best_effort"
    return "retry"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_chunker_graph() -> StateGraph:
    """Construct the Phase 2 semantic chunking sub-graph.

    Flow::

        propose_macro_atoms -> flatten_config -> hoist_state
        -> search_sub_atoms -> critic_validate
        -> [decompose_complex_atoms -> abstract_atoms -> END
            | prepare_chunk_retry -> propose_macro_atoms]

    The ``decompose_complex_atoms`` step recursively splits complex atoms
    into sub-atoms (controlled by ``max_depth`` / ``line_threshold`` in deps).
    The ``abstract_atoms`` step runs the Conceptual Abstraction Agent on
    each atom after the plan is finalized (critic passed or budget exhausted),
    storing domain-agnostic profiles for cross-field semantic retrieval.
    """
    graph = StateGraph(ChunkerState)

    graph.add_node("propose_macro_atoms", propose_macro_atoms)
    graph.add_node("flatten_config", flatten_config)
    graph.add_node("hoist_state", hoist_state)
    graph.add_node("search_sub_atoms", search_sub_atoms)
    graph.add_node("critic_validate", critic_validate)
    graph.add_node("prepare_chunk_retry", prepare_chunk_retry)
    graph.add_node("decompose_complex_atoms", decompose_complex_atoms)
    graph.add_node("abstract_atoms", abstract_atoms)

    graph.set_entry_point("propose_macro_atoms")
    graph.add_edge("propose_macro_atoms", "flatten_config")
    graph.add_edge("flatten_config", "hoist_state")
    graph.add_edge("hoist_state", "search_sub_atoms")
    graph.add_edge("search_sub_atoms", "critic_validate")

    graph.add_conditional_edges(
        "critic_validate",
        route_after_critic,
        {
            "end": "decompose_complex_atoms",
            "end_best_effort": "decompose_complex_atoms",
            "retry": "prepare_chunk_retry",
        },
    )
    graph.add_edge("decompose_complex_atoms", "abstract_atoms")
    graph.add_edge("abstract_atoms", END)
    graph.add_edge("prepare_chunk_retry", "propose_macro_atoms")

    return graph
