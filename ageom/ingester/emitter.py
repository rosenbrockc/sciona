"""Phase 3: Deterministic code generation from a ValidatedMacroPlan.

Generates Pydantic state models, atom wrappers with ``@register_atom``
and ``icontract`` decorators, ghost witness functions, CDGExport nodes
and edges, and pre-filled MatchResults.
"""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Any

from ageom.hunter.llm import LLMClient

from ageom.architect.handoff import CDGExport
from ageom.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from ageom.ingester.models import (
    IngestionBundle,
    MacroAtomSpec,
    ProposedMacroPlan,
    RawDataFlowGraph,
    StateModelSpec,
    ValidatedMacroPlan,
)
from ageom.ingester.prompts import (
    DRAFT_OPAQUE_WITNESS_SYSTEM,
    DRAFT_OPAQUE_WITNESS_USER,
)
from ageom.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationLevel,
    VerificationResult,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Opaque DL boundary witness generation
# ---------------------------------------------------------------------------


def _opaque_witness_fallback(atom: MacroAtomSpec) -> str:
    """Generate a shape-preserving default witness for an opaque atom."""
    fn_name = _snake_case(atom.name)
    witness_name = f"witness_{fn_name}"

    params = []
    for inp in atom.inputs:
        params.append(f"{inp.name}: AbstractArray")
    param_str = ", ".join(params) if params else ""

    lines = [
        f"def {witness_name}({param_str}) -> AbstractArray:",
        f'    """Ghost witness for opaque boundary: {atom.name}."""',
    ]
    if atom.inputs:
        first = atom.inputs[0].name
        lines.append(f"    return AbstractArray(shape={first}.shape, dtype=\"float32\")")
    else:
        lines.append(f'    return AbstractArray(shape=(), dtype="float32")')

    return "\n".join(lines)


async def generate_opaque_witnesses(
    macro_atoms: list[MacroAtomSpec],
    dfg: RawDataFlowGraph,
    llm: LLMClient,
) -> tuple[str, dict[str, str]]:
    """Generate AbstractArray-based witnesses for opaque DL atoms.

    Attempts LLM shape inference; falls back to shape-preserving default.

    Returns (source_code, name_mapping).
    """
    lines = [
        '"""Auto-generated ghost witnesses for opaque DL boundaries."""',
        "",
        "from __future__ import annotations",
        "",
        "try:",
        "    from ageoa.ghost.abstract import AbstractArray",
        "except ImportError:",
        "    pass",
        "",
    ]

    name_map: dict[str, str] = {}

    for atom in macro_atoms:
        if not atom.is_opaque:
            continue

        fn_name = _snake_case(atom.name)
        witness_name = f"witness_{fn_name}"
        name_map[atom.name] = witness_name

        # Attempt LLM-drafted witness
        mf = dfg.methods[0] if dfg.methods else None
        witness_body: str | None = None

        if mf and llm is not None:
            param_specs = ", ".join(
                f'"{p}: AbstractArray"' for p in mf.params
            )
            try:
                user_prompt = DRAFT_OPAQUE_WITNESS_USER.format(
                    class_name=dfg.class_name,
                    base_classes=", ".join(dfg.opaque_base_classes),
                    method_name=mf.name,
                    params=", ".join(mf.params),
                    return_type=mf.return_type or "Any",
                    docstring=mf.docstring or "(none)",
                    fn_name=fn_name,
                    param_specs=param_specs,
                    return_type_spec="AbstractArray",
                )
                response = await llm.complete(
                    DRAFT_OPAQUE_WITNESS_SYSTEM, user_prompt
                )
                raw = json.loads(response)
                witness_body = raw.get("witness_body")
            except Exception as exc:
                logger.warning(
                    "LLM witness drafting failed for %s: %s", atom.name, exc
                )

        if witness_body:
            # Build function with LLM-drafted body
            params = []
            for inp in atom.inputs:
                params.append(f"{inp.name}: AbstractArray")
            param_str = ", ".join(params) if params else ""

            lines.append(f"def {witness_name}({param_str}) -> AbstractArray:")
            lines.append(
                f'    """Ghost witness for opaque boundary: {atom.name}."""'
            )
            # Indent each line of the body
            for body_line in witness_body.strip().splitlines():
                lines.append(f"    {body_line}")
            lines.append("")
        else:
            # Fallback: shape-preserving default
            lines.append(_opaque_witness_fallback(atom))
            lines.append("")

    return "\n".join(lines), name_map


# ---------------------------------------------------------------------------
# State model generation
# ---------------------------------------------------------------------------


def generate_state_models(specs: list[StateModelSpec]) -> str:
    """Generate Pydantic BaseModel classes from state model specs."""
    if not specs:
        return ""

    lines = [
        '"""Auto-generated Pydantic state models for cross-window state."""',
        "",
        "from __future__ import annotations",
        "",
        "from pydantic import BaseModel, Field",
        "",
    ]

    for spec in specs:
        if spec.docstring:
            lines.append(f"class {spec.model_name}(BaseModel):")
            lines.append(f'    """{spec.docstring}"""')
        else:
            lines.append(f"class {spec.model_name}(BaseModel):")

        if not spec.fields:
            lines.append("    pass")
        else:
            for field_name, field_type in spec.fields:
                lines.append(f"    {field_name}: {field_type} = Field(default=None)")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Atom wrapper generation
# ---------------------------------------------------------------------------


def _snake_case(name: str) -> str:
    """Convert a name like 'Signal Conditioner' to 'signal_conditioner'."""
    return name.lower().replace(" ", "_").replace("-", "_")


def generate_atom_wrappers(
    macro_atoms: list[MacroAtomSpec],
    state_models: list[StateModelSpec],
    witness_names: dict[str, str],
) -> str:
    """Generate ``@register_atom`` decorated function wrappers."""
    lines = [
        '"""Auto-generated atom wrappers following the ageoa pattern."""',
        "",
        "from __future__ import annotations",
        "",
        "import icontract",
        "from ageoa.ghost.registry import register_atom",
        "",
    ]

    # Import state models if any
    if state_models:
        model_names = ", ".join(s.model_name for s in state_models)
        lines.append(f"# State models should be imported from the generated state_models module")
        lines.append("")

    # Import witness functions
    if witness_names:
        lines.append(f"# Witness functions should be imported from the generated witnesses module")
        lines.append("")

    for atom in macro_atoms:
        fn_name = _snake_case(atom.name)
        witness_fn = witness_names.get(atom.name, f"witness_{fn_name}")

        # Build parameter list
        params = []
        for inp in atom.inputs:
            params.append(f"{inp.name}: {inp.type_desc}")
        param_str = ", ".join(params) if params else ""

        # Build return type
        if atom.outputs:
            if len(atom.outputs) == 1:
                ret_type = atom.outputs[0].type_desc
            else:
                ret_type = "tuple[" + ", ".join(o.type_desc for o in atom.outputs) + "]"
        else:
            ret_type = "None"

        lines.append(f"@register_atom({witness_fn})")
        lines.append(f"def {fn_name}({param_str}) -> {ret_type}:")
        if atom.description:
            lines.append(f'    """{atom.description}"""')
        lines.append(f"    raise NotImplementedError(\"Wire to original implementation\")")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stateful wrapper generation
# ---------------------------------------------------------------------------


def generate_stateful_wrappers(
    macro_atoms: list[MacroAtomSpec],
    state_models: list[StateModelSpec],
    class_name: str,
    witness_names: dict[str, str],
) -> str:
    """Generate ``@register_atom`` wrappers with inject/run/extract state pattern.

    Each wrapper instantiates the legacy class via ``__new__``, injects ALL
    state fields, runs the original method(s), and extracts ALL state fields
    back into an immutable ``model_copy`` update.
    """
    if not state_models:
        return generate_atom_wrappers(macro_atoms, state_models, witness_names)

    state_model = state_models[0]
    state_type = state_model.model_name
    fields = state_model.fields  # list of (field_name, field_type)

    lines = [
        '"""Auto-generated stateful atom wrappers following the ageoa pattern."""',
        "",
        "from __future__ import annotations",
        "",
        "import icontract",
        "from ageoa.ghost.registry import register_atom",
        "",
        f"# Import the original class for __new__ instantiation",
        f"# from <source_module> import {class_name}",
        "",
        f"# State model should be imported from the generated state_models module",
        f"# from <state_module> import {state_type}",
        "",
    ]

    # Import witness functions
    if witness_names:
        lines.append("# Witness functions should be imported from the generated witnesses module")
        lines.append("")

    for atom in macro_atoms:
        fn_name = _snake_case(atom.name)
        witness_fn = witness_names.get(atom.name, f"witness_{fn_name}")

        # Build parameter list — original params + state
        params = []
        for inp in atom.inputs:
            params.append(f"{inp.name}: {inp.type_desc}")
        params.append(f"state: {state_type}")
        param_str = ", ".join(params)

        # Build return type — (original_return, StateType)
        if atom.outputs:
            if len(atom.outputs) == 1:
                orig_ret = atom.outputs[0].type_desc
            else:
                orig_ret = "tuple[" + ", ".join(o.type_desc for o in atom.outputs) + "]"
        else:
            orig_ret = "None"
        ret_type = f"tuple[{orig_ret}, {state_type}]"

        lines.append(f"@register_atom({witness_fn})")
        lines.append(f"def {fn_name}({param_str}) -> {ret_type}:")
        if atom.description:
            lines.append(f'    """Stateless wrapper: Functional Core, Imperative Shell.')
            lines.append(f"")
            lines.append(f"    {atom.description}")
            lines.append(f'    """')
        else:
            lines.append(f'    """Stateless wrapper: Functional Core, Imperative Shell."""')

        # Instantiate via __new__
        lines.append(f"    obj = {class_name}.__new__({class_name})")

        # Inject ALL state fields
        for field_name, _ in fields:
            lines.append(f"    obj.{field_name} = state.{field_name}")

        # Run method(s)
        for method_name in atom.method_names:
            if method_name == "__init__":
                continue
            # Build call args from atom inputs
            call_args = ", ".join(inp.name for inp in atom.inputs)
            lines.append(f"    obj.{method_name}({call_args})")

        # Extract ALL state fields via model_copy
        update_entries = ", ".join(
            f'"{fname}": obj.{fname}' for fname, _ in fields
        )
        lines.append(f"    new_state = state.model_copy(update={{")
        for fname, _ in fields:
            lines.append(f'        "{fname}": obj.{fname},')
        lines.append(f"    }})")

        # Build return value
        if atom.outputs:
            if len(atom.outputs) == 1:
                out = atom.outputs[0]
                lines.append(f"    result = obj.{out.name}")
                lines.append(f"    return result, new_state")
            else:
                out_names = ", ".join(f"obj.{o.name}" for o in atom.outputs)
                lines.append(f"    result = ({out_names})")
                lines.append(f"    return result, new_state")
        else:
            lines.append(f"    return None, new_state")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ghost witness generation
# ---------------------------------------------------------------------------


def generate_ghost_witnesses(
    macro_atoms: list[MacroAtomSpec],
    state_models: list[StateModelSpec] | None = None,
) -> tuple[str, dict[str, str]]:
    """Generate ghost witness functions.

    Returns (source_code, name_mapping) where name_mapping maps
    atom name -> witness function name.

    When *state_models* is non-empty, each witness gains a
    ``state: AbstractSignal`` parameter and returns
    ``tuple[AbstractSignal, AbstractSignal]`` (result, state pass-through).
    """
    has_state = bool(state_models)

    lines = [
        '"""Auto-generated ghost witness functions for abstract simulation."""',
        "",
        "from __future__ import annotations",
        "",
        "try:",
        "    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar",
        "except ImportError:",
        "    pass",
        "",
    ]

    name_map: dict[str, str] = {}

    for atom in macro_atoms:
        fn_name = _snake_case(atom.name)
        witness_name = f"witness_{fn_name}"
        name_map[atom.name] = witness_name

        # Build parameter list
        params = []
        for inp in atom.inputs:
            params.append(f"{inp.name}: AbstractSignal")
        if has_state:
            params.append("state: AbstractSignal")
        param_str = ", ".join(params) if params else ""

        # Return type
        if has_state:
            if atom.outputs:
                ret_type = "tuple[AbstractSignal, AbstractSignal]"
            else:
                ret_type = "tuple[None, AbstractSignal]"
        else:
            if atom.outputs:
                ret_type = "AbstractSignal"
            else:
                ret_type = "None"

        lines.append(f"def {witness_name}({param_str}) -> {ret_type}:")
        lines.append(f'    """Ghost witness for {atom.name}."""')

        if atom.inputs and atom.outputs:
            first_input = atom.inputs[0].name
            lines.append(f"    result = AbstractSignal(")
            lines.append(f"        shape={first_input}.shape,")
            lines.append(f'        dtype="float64",')
            lines.append(f"        sampling_rate=getattr({first_input}, 'sampling_rate', 44100.0),")
            lines.append(f'        domain="time",')
            lines.append(f"    )")
            if has_state:
                lines.append(f"    return result, state")
            else:
                lines.append(f"    return result")
        else:
            if has_state:
                lines.append(f"    return None, state")
            else:
                lines.append(f"    return None")
        lines.append("")

    return "\n".join(lines), name_map


# ---------------------------------------------------------------------------
# CDG construction
# ---------------------------------------------------------------------------


def build_cdg_export(
    plan: ValidatedMacroPlan, class_name: str
) -> CDGExport:
    """Build a CDGExport with root DECOMPOSED node + ATOMIC children."""
    root_node = AlgorithmicNode(
        node_id=f"{class_name}_root",
        name=class_name,
        description=f"Ingested pipeline from {class_name}",
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.DECOMPOSED,
        children=[_snake_case(a.name) for a in plan.plan.macro_atoms],
        depth=0,
    )

    child_nodes = []
    for atom in plan.plan.macro_atoms:
        node_id = _snake_case(atom.name)
        child = AlgorithmicNode(
            node_id=node_id,
            parent_id=root_node.node_id,
            name=atom.name,
            description=atom.description,
            concept_type=atom.concept_type,
            inputs=list(atom.inputs),
            outputs=list(atom.outputs),
            status=NodeStatus.ATOMIC,
            is_optional=atom.is_optional,
            is_opaque=atom.is_opaque,
            type_signature=_build_type_signature(atom),
            depth=1,
        )
        child_nodes.append(child)

    # Build typed edges
    edges = []
    for edge_def in plan.plan.edge_definitions:
        edges.append(DependencyEdge(
            source_id=edge_def.source_id,
            target_id=edge_def.target_id,
            output_name=edge_def.output_name,
            input_name=edge_def.input_name,
            source_type=edge_def.source_type,
            target_type=edge_def.target_type,
        ))

    all_nodes = [root_node] + child_nodes
    return CDGExport(
        nodes=all_nodes,
        edges=edges,
        metadata={
            "source": "ingester",
            "class_name": class_name,
        },
    )


def _build_type_signature(atom: MacroAtomSpec) -> str:
    """Build a Python type signature string from IOSpec."""
    inputs = ", ".join(f"{i.name}: {i.type_desc}" for i in atom.inputs)
    if atom.outputs:
        if len(atom.outputs) == 1:
            ret = atom.outputs[0].type_desc
        else:
            ret = "tuple[" + ", ".join(o.type_desc for o in atom.outputs) + "]"
    else:
        ret = "None"
    return f"({inputs}) -> {ret}"


# ---------------------------------------------------------------------------
# Sub-graph construction
# ---------------------------------------------------------------------------


def build_sub_graphs(plan: ValidatedMacroPlan) -> dict[str, CDGExport]:
    """Build zoom-in sub-graphs from sub_atom_refs."""
    sub_graphs: dict[str, CDGExport] = {}

    for atom in plan.plan.macro_atoms:
        # Find sub-atom refs relevant to this macro-atom
        relevant_refs = [
            ref for ref in plan.plan.sub_atom_refs
            if ref.similarity_score > 0.5
        ]
        if not relevant_refs:
            continue

        node_id = _snake_case(atom.name)
        root = AlgorithmicNode(
            node_id=f"{node_id}_sub_root",
            name=f"{atom.name} (sub-graph)",
            description=f"Zoom-in decomposition of {atom.name}",
            concept_type=atom.concept_type,
            status=NodeStatus.DECOMPOSED,
            children=[_snake_case(r.atom_name) for r in relevant_refs],
            depth=0,
        )

        children = []
        for ref in relevant_refs:
            child = AlgorithmicNode(
                node_id=_snake_case(ref.atom_name),
                parent_id=root.node_id,
                name=ref.atom_name,
                description=f"Existing atom (similarity: {ref.similarity_score:.2f})",
                concept_type=atom.concept_type,
                status=NodeStatus.ATOMIC,
                depth=1,
            )
            children.append(child)

        sub_graphs[node_id] = CDGExport(
            nodes=[root] + children,
            edges=[],
            metadata={"parent_atom": atom.name},
        )

    return sub_graphs


# ---------------------------------------------------------------------------
# Match results
# ---------------------------------------------------------------------------


def build_match_results(
    cdg: CDGExport, atoms_source: str
) -> list[MatchResult]:
    """Build pre-filled MatchResults with verified=True for atomic leaves."""
    results = []
    for node in cdg.nodes:
        if node.status != NodeStatus.ATOMIC:
            continue

        fn_name = _snake_case(node.name)
        decl = Declaration(
            name=fn_name,
            type_signature=node.type_signature,
            docstring=node.description,
            source_lib="ingester",
            prover=Prover.PYTHON,
            raw_code="",
        )
        candidate = CandidateMatch(
            declaration=decl,
            score=1.0,
            retrieval_method="ingester",
        )
        vr = VerificationResult(
            candidate=candidate,
            verified=True,
            verification_level=VerificationLevel.TYPE_CHECKED,
        )
        pdg_node = PDGNode(
            predicate_id=node.node_id,
            statement=node.type_signature,
            informal_desc=node.description,
            prover=Prover.PYTHON,
        )
        results.append(MatchResult(
            pdg_node=pdg_node,
            verified_match=vr,
            all_candidates=[candidate],
            all_verifications=[vr],
        ))

    return results


# ---------------------------------------------------------------------------
# Procedural plan builder (bypasses Phase 2)
# ---------------------------------------------------------------------------


def _title_case(name: str) -> str:
    """Convert a snake_case name like 'remove_baseline' to 'Remove Baseline'."""
    return name.replace("_", " ").title()


def build_procedural_plan(
    dfg: RawDataFlowGraph, pipeline_name: str
) -> ValidatedMacroPlan:
    """Build a ValidatedMacroPlan from procedural SSA edges (no LLM needed).

    Each top-level function becomes one MacroAtomSpec.  Edges come directly
    from ``dfg.inferred_edges`` computed by the SSA visitor.
    """
    macro_atoms: list[MacroAtomSpec] = []
    for mf in dfg.methods:
        inputs = [IOSpec(name=p, type_desc="Any") for p in mf.params]
        outputs = (
            [IOSpec(name="result", type_desc=mf.return_type or "Any")]
            if mf.return_type
            else [IOSpec(name="result", type_desc="Any")]
        )
        macro_atoms.append(MacroAtomSpec(
            name=_title_case(mf.name),
            description=mf.docstring,
            method_names=[mf.name],
            inputs=inputs,
            outputs=outputs,
            concept_type=ConceptType.CUSTOM,
        ))

    plan = ProposedMacroPlan(
        macro_atoms=macro_atoms,
        edge_definitions=list(dfg.inferred_edges),
    )

    return ValidatedMacroPlan(plan=plan, all_attrs_accounted=True)


# ---------------------------------------------------------------------------
# Top-level emitter
# ---------------------------------------------------------------------------


def emit_ingestion_bundle(
    plan: ValidatedMacroPlan,
    class_name: str,
    source_file: str = "",
) -> IngestionBundle:
    """Assemble all Phase 3 outputs into an IngestionBundle."""
    # Check for opaque atoms — use fallback witnesses (synchronous)
    has_opaque = any(a.is_opaque for a in plan.plan.macro_atoms)

    if has_opaque:
        # Generate opaque witness stubs (fallback, no LLM)
        witness_lines = [
            '"""Auto-generated ghost witnesses for opaque DL boundaries."""',
            "",
            "from __future__ import annotations",
            "",
            "try:",
            "    from ageoa.ghost.abstract import AbstractArray",
            "except ImportError:",
            "    pass",
            "",
        ]
        witness_names: dict[str, str] = {}
        for atom in plan.plan.macro_atoms:
            if atom.is_opaque:
                witness_lines.append(_opaque_witness_fallback(atom))
                witness_lines.append("")
                fn_name = _snake_case(atom.name)
                witness_names[atom.name] = f"witness_{fn_name}"
        witness_source = "\n".join(witness_lines)
    else:
        # Generate witnesses first (need name mapping for atoms)
        witness_source, witness_names = generate_ghost_witnesses(
            plan.plan.macro_atoms,
            state_models=plan.plan.state_models,
        )

    # Generate state models
    state_model_source = generate_state_models(plan.plan.state_models)

    # Generate atom wrappers — stateful if state models exist
    if plan.plan.state_models:
        atoms_source = generate_stateful_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            class_name,
            witness_names,
        )
    else:
        atoms_source = generate_atom_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            witness_names,
        )

    # Build CDG
    cdg = build_cdg_export(plan, class_name)

    # Build sub-graphs
    sub_graphs = build_sub_graphs(plan)

    # Build match results
    match_results = build_match_results(cdg, atoms_source)

    return IngestionBundle(
        cdg=cdg,
        sub_graphs=sub_graphs,
        generated_atoms=atoms_source,
        generated_state_models=state_model_source,
        generated_witnesses=witness_source,
        match_results=match_results,
    )
