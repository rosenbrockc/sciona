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
    ConceptualProfile,
    IngestionBundle,
    MacroAtomSpec,
    ProposedMacroPlan,
    RawDataFlowGraph,
    StateModelSpec,
    StochasticTraceSpec,
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

# Bayesian concept types that get specialized witness templates
_BAYESIAN_CONCEPT_TYPES = frozenset({
    ConceptType.SAMPLER,
    ConceptType.LOG_PROB,
    ConceptType.POSTERIOR_UPDATE,
    ConceptType.CONJUGATE_UPDATE,
    ConceptType.VARIATIONAL_INFERENCE,
    ConceptType.PRIOR_INIT,
})


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
        "import torch",
        "import jax",
        "import jax.numpy as jnp",
        "import haiku as hk",
        "",
        "import networkx as nx  # type: ignore",
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
                from ageom.llm_router import INGESTER_OPAQUE_WITNESS, select_llm

                response = await select_llm(llm, INGESTER_OPAQUE_WITNESS).complete(
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
    """Generate Pydantic BaseModel classes from state model specs.

    When a spec has a ``stochastic`` field, injects RNG key and MCMC trace
    fields with appropriate types and defaults.
    """
    if not specs:
        return ""

    has_stochastic = any(s.stochastic is not None for s in specs)

    lines = [
        '"""Auto-generated Pydantic state models for cross-window state."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Any",
        "",
        "import torch",
        "import jax",
        "import jax.numpy as jnp",
        "import haiku as hk",
        "",
        "import networkx as nx  # type: ignore",
        "",
        "from pydantic import BaseModel, ConfigDict, Field",
        "",
    ]

    if has_stochastic:
        lines.extend([
            "import numpy as np",
            "",
        ])

    for spec in specs:
        if spec.docstring:
            lines.append(f"class {spec.model_name}(BaseModel):")
            lines.append(f'    """{spec.docstring}"""')
        else:
            lines.append(f"class {spec.model_name}(BaseModel):")

        lines.append("    model_config = ConfigDict(arbitrary_types_allowed=True)")
        lines.append("")
        if not spec.fields and spec.stochastic is None:
            lines.append("    pass")
        else:
            for field_name, field_type in spec.fields:
                lines.append(f"    {field_name}: {field_type} | None = Field(default=None)")

            # Inject stochastic state fields
            if spec.stochastic is not None:
                st = spec.stochastic
                lines.append("")
                lines.append("    # --- Stochastic state (auto-generated) ---")
                lines.append(f"    {st.rng_field}: Any = Field(")
                lines.append(f"        default=None,")
                lines.append(f'        description="RNG state ({st.rng_type}). '
                             f'Split before each stochastic atom.",')
                lines.append(f"    )")

                if st.trace_field:
                    dims_str = str(st.trace_param_dims)
                    lines.append(f"    {st.trace_field}: Any = Field(")
                    lines.append(f"        default=None,")
                    lines.append(f'        description="MCMC trace. '
                                 f'param_dims={dims_str}, '
                                 f'chains={st.chain_count}, '
                                 f'warmup={st.warmup_steps}",')
                    lines.append(f"    )")
                    lines.append(f"    mcmc_step_count: int = Field(default=0)")
                    lines.append(f"    mcmc_accept_rate: float = Field(default=0.0)")
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
    source_language: str = "python",
) -> str:
    """Generate ``@register_atom`` decorated function wrappers."""
    lines = [
        '"""Auto-generated atom wrappers following the ageoa pattern."""',
        "",
        "from __future__ import annotations",
        "",
        "import torch",
        "import jax",
        "import jax.numpy as jnp",
        "import haiku as hk",
        "",
        "import networkx as nx  # type: ignore",
        "import icontract",
        "from ageoa.ghost.registry import register_atom",
        "",
    ]

    # Add FFI imports for non-Python sources
    if source_language != "python":
        from ageom.ingester.ffi_emitter import generate_ffi_imports
        lines.append(generate_ffi_imports(source_language))
        lines.append("")

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

        for deco in getattr(atom, "decorators", []):
            lines.append(deco)
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
    source_language: str = "python",
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
        "import torch",
        "import jax",
        "import jax.numpy as jnp",
        "import haiku as hk",
        "",
        "import networkx as nx  # type: ignore",
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

        for deco in getattr(atom, "decorators", []):
            lines.append(deco)
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


def _generate_bayesian_witness(
    atom: MacroAtomSpec,
    fn_name: str,
    witness_name: str,
    has_state: bool,
) -> list[str]:
    """Generate a specialized witness for a Bayesian atom.

    Routes to the appropriate witness template based on concept_type.
    """
    lines: list[str] = []
    ct = atom.concept_type

    if ct == ConceptType.PRIOR_INIT:
        params = ["event_shape: tuple[int, ...]", "family: str = \"normal\""]
        lines.append(f"def {witness_name}({', '.join(params)}) -> AbstractDistribution:")
        lines.append(f'    """Ghost witness for prior init: {atom.name}."""')
        lines.append(f"    return AbstractDistribution(")
        lines.append(f"        family=family,")
        lines.append(f"        event_shape=event_shape,")
        lines.append(f"    )")

    elif ct == ConceptType.LOG_PROB:
        params = ["dist: AbstractDistribution", "samples: AbstractArray"]
        lines.append(f"def {witness_name}({', '.join(params)}) -> AbstractScalar:")
        lines.append(f'    """Ghost witness for log-prob: {atom.name}."""')
        lines.append(f"    n_event = len(dist.event_shape)")
        lines.append(f"    if n_event > 0:")
        lines.append(f"        sample_tail = samples.shape[-n_event:]")
        lines.append(f"        if sample_tail != dist.event_shape:")
        lines.append(f"            raise ValueError(")
        lines.append(f'                f"Sample dims {{sample_tail}} vs event_shape {{dist.event_shape}}"')
        lines.append(f"            )")
        lines.append(f'    return AbstractScalar(dtype="float64", max_val=0.0)')

    elif ct == ConceptType.SAMPLER:
        params = [
            "trace: AbstractMCMCTrace",
            "target: AbstractDistribution",
            "rng: AbstractRNGState",
        ]
        ret = "tuple[AbstractMCMCTrace, AbstractRNGState]"
        lines.append(f"def {witness_name}({', '.join(params)}) -> {ret}:")
        lines.append(f'    """Ghost witness for MCMC sampler: {atom.name}."""')
        lines.append(f"    if trace.param_dims != target.event_shape:")
        lines.append(f"        raise ValueError(")
        lines.append(f'            f"param_dims {{trace.param_dims}} vs '
                      f'event_shape {{target.event_shape}}"')
        lines.append(f"        )")
        lines.append(f"    return trace.step(accepted=True), rng.advance(n_draws=1)")

    elif ct == ConceptType.POSTERIOR_UPDATE:
        params = [
            "prior: AbstractDistribution",
            "likelihood: AbstractDistribution",
            "data_shape: tuple[int, ...]",
        ]
        lines.append(f"def {witness_name}({', '.join(params)}) -> AbstractDistribution:")
        lines.append(f'    """Ghost witness for posterior update: {atom.name}."""')
        lines.append(f"    prior.assert_conjugate_to(likelihood)")
        lines.append(f"    return AbstractDistribution(")
        lines.append(f"        family=prior.family,")
        lines.append(f"        event_shape=prior.event_shape,")
        lines.append(f"        batch_shape=prior.batch_shape,")
        lines.append(f"        support_lower=prior.support_lower,")
        lines.append(f"        support_upper=prior.support_upper,")
        lines.append(f"        is_discrete=prior.is_discrete,")
        lines.append(f"    )")

    elif ct == ConceptType.CONJUGATE_UPDATE:
        params = [
            "prior: AbstractDistribution",
            "sufficient_stats: AbstractArray",
        ]
        lines.append(f"def {witness_name}({', '.join(params)}) -> AbstractDistribution:")
        lines.append(f'    """Ghost witness for closed-form conjugate update: {atom.name}."""')
        lines.append(f"    # Closed-form update: no sampling trace or RNG threading required.")
        lines.append(f"    return AbstractDistribution(")
        lines.append(f"        family=prior.family,")
        lines.append(f"        event_shape=prior.event_shape,")
        lines.append(f"        batch_shape=prior.batch_shape,")
        lines.append(f"        support_lower=prior.support_lower,")
        lines.append(f"        support_upper=prior.support_upper,")
        lines.append(f"        is_discrete=prior.is_discrete,")
        lines.append(f"    )")

    elif ct == ConceptType.VARIATIONAL_INFERENCE:
        params = [
            "q_dist: AbstractDistribution",
            "p_dist: AbstractDistribution",
            "n_samples: int = 1",
        ]
        lines.append(f"def {witness_name}({', '.join(params)}) -> AbstractScalar:")
        lines.append(f'    """Ghost witness for VI ELBO: {atom.name}."""')
        lines.append(f"    if q_dist.event_shape != p_dist.event_shape:")
        lines.append(f"        raise ValueError(")
        lines.append(f'            f"q event_shape {{q_dist.event_shape}} vs '
                      f'p event_shape {{p_dist.event_shape}}"')
        lines.append(f"        )")
        lines.append(f'    return AbstractScalar(dtype="float64")')

    lines.append("")
    return lines


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

    Bayesian atoms (concept_type in SAMPLER, LOG_PROB, POSTERIOR_UPDATE,
    VARIATIONAL_INFERENCE, PRIOR_INIT) get specialized witnesses that use
    AbstractDistribution, AbstractRNGState, and AbstractMCMCTrace.
    """
    has_state = bool(state_models)
    has_bayesian = any(a.concept_type in _BAYESIAN_CONCEPT_TYPES for a in macro_atoms)
    has_sampler = any(a.concept_type == ConceptType.SAMPLER for a in macro_atoms)

    lines = [
        '"""Auto-generated ghost witness functions for abstract simulation."""',
        "",
        "from __future__ import annotations",
        "",
        "import torch",
        "import jax",
        "import jax.numpy as jnp",
        "import haiku as hk",
        "",
        "import networkx as nx  # type: ignore",
        "",
        "try:",
        "    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar",
    ]
    if has_bayesian:
        lines.extend([
            "    from ageoa.ghost.abstract import AbstractDistribution",
        ])
    if has_sampler:
        lines.extend([
            "    from ageoa.ghost.abstract import AbstractMCMCTrace",
            "    from ageoa.ghost.abstract import AbstractRNGState",
        ])
    lines.extend([
        "except ImportError:",
        "    pass",
        "",
    ])

    name_map: dict[str, str] = {}

    for atom in macro_atoms:
        if atom.is_opaque:
            continue
        fn_name = _snake_case(atom.name)
        witness_name = f"witness_{fn_name}"
        name_map[atom.name] = witness_name

        # Bayesian atoms get specialized witness templates
        if atom.concept_type in _BAYESIAN_CONCEPT_TYPES:
            lines.extend(_generate_bayesian_witness(
                atom, fn_name, witness_name, has_state
            ))
            continue

        # Default DSP/generic witness
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
# Conceptual profile → plain-text summary
# ---------------------------------------------------------------------------


def _profile_to_summary(profile: ConceptualProfile | None) -> str:
    """Convert a ConceptualProfile to a plain-text summary for embedding."""
    if not profile or not profile.abstract_name:
        return ""
    parts = [profile.abstract_name]
    if profile.conceptual_transform:
        parts.append(profile.conceptual_transform)
    if profile.cross_disciplinary_applications:
        parts.append("Applications: " + ", ".join(profile.cross_disciplinary_applications))
    return ". ".join(parts)


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
            is_external=getattr(atom, "is_external", False),
            type_signature=_build_type_signature(atom),
            conceptual_summary=_profile_to_summary(atom.conceptual_profile),
            depth=1,
        )
        child_nodes.append(child)

    # Build typed edges
    edges = []
    for edge_def in plan.plan.edge_definitions:
        edges.append(DependencyEdge(
            source_id=_snake_case(_title_case(edge_def.source_id)),
            target_id=_snake_case(_title_case(edge_def.target_id)),
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
            conceptual_summary=node.conceptual_summary,
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
            decorators=mf.decorators,
            is_external=mf.is_external,
            concept_type=ConceptType.EXTERNAL_TOOL if mf.is_external else ConceptType.CUSTOM,
            name=_title_case(mf.name),
            description=mf.docstring,
            method_names=[mf.name],
            inputs=inputs,
            outputs=outputs,
        ))

    plan = ProposedMacroPlan(
        macro_atoms=macro_atoms,
        edge_definitions=list(dfg.inferred_edges),
    )

    return ValidatedMacroPlan(plan=plan, all_attrs_accounted=True)


def _linearize_conjugate_sequence(plan: ValidatedMacroPlan) -> ValidatedMacroPlan:
    """Ensure conjugate updates follow data->update->distribution edges."""
    atoms = plan.plan.macro_atoms
    if not atoms:
        return plan

    edges = list(plan.plan.edge_definitions)
    seen = {
        (
            e.source_id, e.target_id, e.output_name, e.input_name,
            e.source_type, e.target_type,
        )
        for e in edges
    }

    def pick_output(atom: MacroAtomSpec) -> tuple[str, str]:
        if atom.outputs:
            pref = next(
                (o for o in atom.outputs if any(
                    h in o.name.lower() for h in ("data", "obs", "sample", "stats", "posterior", "params")
                )),
                atom.outputs[0],
            )
            return pref.name, pref.type_desc
        return "result", "Any"

    def pick_input(atom: MacroAtomSpec, out_name: str) -> tuple[str, str]:
        if atom.inputs:
            for inp in atom.inputs:
                if inp.name == out_name:
                    return inp.name, inp.type_desc
            pref = next(
                (i for i in atom.inputs if any(
                    h in i.name.lower() for h in ("data", "obs", "sample", "stats", "posterior", "params")
                )),
                atom.inputs[0],
            )
            return pref.name, pref.type_desc
        return out_name, "Any"

    def add_edge(edge: DependencyEdge) -> None:
        key = (
            edge.source_id, edge.target_id, edge.output_name, edge.input_name,
            edge.source_type, edge.target_type,
        )
        if key not in seen:
            seen.add(key)
            edges.append(edge)

    for conj in [a for a in atoms if a.concept_type == ConceptType.CONJUGATE_UPDATE]:
        conj_id = _snake_case(conj.name)
        incoming = [e for e in edges if e.target_id == conj_id]
        outgoing = [e for e in edges if e.source_id == conj_id]

        if not incoming:
            data_atom = next(
                (
                    a for a in atoms
                    if a.concept_type != ConceptType.CONJUGATE_UPDATE
                    and _snake_case(a.name) != conj_id
                    and (
                        "data" in a.name.lower()
                        or "ingest" in a.name.lower()
                        or any(h in o.name.lower() for o in a.outputs for h in ("data", "obs", "sample", "stats"))
                    )
                ),
                None,
            )
            if data_atom is not None:
                out_name, out_type = pick_output(data_atom)
                in_name, in_type = pick_input(conj, out_name)
                add_edge(DependencyEdge(
                    source_id=_snake_case(data_atom.name),
                    target_id=conj_id,
                    output_name=out_name,
                    input_name=in_name,
                    source_type=out_type,
                    target_type=in_type,
                ))

        if not outgoing:
            dist_atom = next(
                (
                    a for a in atoms
                    if _snake_case(a.name) != conj_id
                    and (
                        a.concept_type in {ConceptType.PRIOR_DISTRIBUTION, ConceptType.PRIOR_INIT}
                        or "distribution" in a.name.lower()
                        or "posterior" in a.name.lower()
                        or "construct" in a.name.lower()
                    )
                ),
                None,
            )
            if dist_atom is not None:
                out_name, out_type = pick_output(conj)
                in_name, in_type = pick_input(dist_atom, out_name)
                add_edge(DependencyEdge(
                    source_id=conj_id,
                    target_id=_snake_case(dist_atom.name),
                    output_name=out_name,
                    input_name=in_name,
                    source_type=out_type,
                    target_type=in_type,
                ))

    updated = plan.plan.model_copy(update={"edge_definitions": edges})
    return plan.model_copy(update={"plan": updated})


# ---------------------------------------------------------------------------
# Top-level emitter
# ---------------------------------------------------------------------------


def emit_ingestion_bundle(
    plan: ValidatedMacroPlan,
    class_name: str,
    source_file: str = "",
    source_language: str = "python",
) -> IngestionBundle:
    """Assemble all Phase 3 outputs into an IngestionBundle."""
    # Conjugate updates should follow deterministic
    # data->hyperparameter update->distribution construction flow.
    plan = _linearize_conjugate_sequence(plan)

    # Check for opaque atoms
    has_opaque = any(a.is_opaque for a in plan.plan.macro_atoms)

    # Generate witnesses first (need name mapping for atoms)
    witness_source, witness_names = generate_ghost_witnesses(
        plan.plan.macro_atoms,
        state_models=plan.plan.state_models,
    )

    if has_opaque:
        # Append opaque witness stubs (fallback, no LLM)
        opaque_lines = [
            "",
            "# Opaque DL boundaries",
            "",
        ]
        for atom in plan.plan.macro_atoms:
            if atom.is_opaque:
                opaque_lines.append(_opaque_witness_fallback(atom))
                opaque_lines.append("")
                fn_name = _snake_case(atom.name)
                witness_names[atom.name] = f"witness_{fn_name}"
        witness_source += "\n".join(opaque_lines)

    # Generate state models
    state_model_source = generate_state_models(plan.plan.state_models)

    # Generate atom wrappers — stateful if state models exist
    if plan.plan.state_models:
        atoms_source = generate_stateful_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            class_name,
            witness_names,
            source_language=source_language,
        )
    else:
        atoms_source = generate_atom_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            witness_names,
            source_language=source_language,
        )

    # Append FFI binding stubs for non-Python sources
    if source_language != "python":
        from ageom.ingester.ffi_emitter import generate_ffi_bindings
        ffi_source = generate_ffi_bindings(
            plan.plan.macro_atoms, source_language
        )
        atoms_source = atoms_source + "\n\n" + ffi_source

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
