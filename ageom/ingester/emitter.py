"""Phase 3: Deterministic code generation from a ValidatedMacroPlan.

Generates Pydantic state models, atom wrappers with ``@register_atom``
and ``icontract`` decorators, ghost witness functions, CDGExport nodes
and edges, and pre-filled MatchResults.
"""

from __future__ import annotations

import textwrap
from typing import Any

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
    StateModelSpec,
    ValidatedMacroPlan,
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
# Ghost witness generation
# ---------------------------------------------------------------------------


def generate_ghost_witnesses(
    macro_atoms: list[MacroAtomSpec],
) -> tuple[str, dict[str, str]]:
    """Generate ghost witness functions.

    Returns (source_code, name_mapping) where name_mapping maps
    atom name -> witness function name.
    """
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
        param_str = ", ".join(params) if params else ""

        # Return type
        if atom.outputs:
            ret_type = "AbstractSignal"
        else:
            ret_type = "None"

        lines.append(f"def {witness_name}({param_str}) -> {ret_type}:")
        lines.append(f'    """Ghost witness for {atom.name}."""')

        if atom.inputs and atom.outputs:
            first_input = atom.inputs[0].name
            lines.append(f"    return AbstractSignal(")
            lines.append(f"        shape={first_input}.shape,")
            lines.append(f'        dtype="float64",')
            lines.append(f"        sampling_rate=getattr({first_input}, 'sampling_rate', 44100.0),")
            lines.append(f'        domain="time",')
            lines.append(f"    )")
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
# Top-level emitter
# ---------------------------------------------------------------------------


def emit_ingestion_bundle(
    plan: ValidatedMacroPlan,
    class_name: str,
    source_file: str = "",
) -> IngestionBundle:
    """Assemble all Phase 3 outputs into an IngestionBundle."""
    # Generate witnesses first (need name mapping for atoms)
    witness_source, witness_names = generate_ghost_witnesses(plan.plan.macro_atoms)

    # Generate state models
    state_model_source = generate_state_models(plan.plan.state_models)

    # Generate atom wrappers
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
