"""LLM prompt templates for the Smart Ingester.

Each prompt pair (SYSTEM / USER) maps to a specific phase and expected
JSON output schema.
"""

from __future__ import annotations

from sciona.architect.models import ConceptType

_CONCEPT_TYPE_LIST = ", ".join(
    ct.value for ct in ConceptType if ct != ConceptType.CUSTOM
) + ", custom"

# ---------------------------------------------------------------------------
# Phase 2: Semantic chunking
# ---------------------------------------------------------------------------

SEMANTIC_CHUNK_SYSTEM = """\
You are an expert software architect converting stateful classes into \
pure functional algorithm graphs.

Given method summaries, self.* attribute access, and config-gated branches, \
group methods into MacroAtomSpecs named by intent, not implementation.

Core rules:
1. Every method appears in exactly one macro-atom.
2. Private helpers (_foo) are grouped with their caller.
3. __init__ setup may be its own atom or grouped with the first stage.
4. Config branches become optional nodes (is_optional=true).
5. Choose concept_type from: """ + _CONCEPT_TYPE_LIST + """.

Orchestration / data-flow type guidance:
- state_init — initializes, resets, or bootstraps state/containers (no computation).
- data_assembly — builds, assembles, prepares, or materializes composite data structures.
- conditional_routing — selects, gates, guards, or dispatches based on conditions.
- data_extraction — fetches, loads, reads, or parses external data sources.
- visualization — plots, renders, or draws visual output.
- observability — emits debug info, logs, traces, or records diagnostics.

Bayesian/state-space detection requirements (MANDATORY):
1. Detect distinct state-space structures and name them explicitly: \
   covariance (P, Q, R), latent mean/variance, particles, weights, \
   ancestors, momenta, mass matrix, trace, RNG state, PRNGKey.
2. Any structure that persists across calls MUST be treated as immutable \
   flowing state, never hidden mutation. If an atom both reads and writes \
   one of these values, model it as state_in -> state_out with a new object.
3. Transition kernels must be pure functions. For Kalman-like algorithms, \
   force decomposition into Predict and Update kernels (or equivalent) where \
   each consumes a StateModelSpec and returns a brand new StateModelSpec.
4. Particle filter flow must be explicit (e.g., propagate/propose -> \
   reweight -> resample) with particles/weights threaded as immutable state.
5. JAX functional purity is the canonical stochastic pattern: thread \
   ``jax.random.PRNGKey`` explicitly as input and output (split keys), never \
   as implicit global state.

Bayesian / probabilistic inference patterns:
- **prior_init**: initialize prior distributions or hyperparameter priors \
  (Normal/Dirichlet, mu_0/sigma_0, etc.).
- **log_prob** / **likelihood_evaluation**: evaluate log-probability, \
  log_likelihood, score, density.
- **sampler** / **mcmc_kernel** / **mcmc_proposal**: stochastic transition \
  steps (Metropolis-Hastings accept/reject, HMC leapfrog, Gibbs).
- **posterior_update** / **smc_reweight**: conjugate updates, Kalman update, \
  particle weight update and normalization.
- **variational_inference** / **vi_elbo**: ELBO/KL/reparameterization updates.

Oracle isolation requirements:
1. If method metadata has ``is_oracle=True`` or behavior is stateless \
   log-density/gradient/likelihood evaluation, isolate it as a pure oracle \
   atom (no persistent state writes).
2. Oracle outputs must be explicit (log_prob, gradient, likelihood) and fed \
   by explicit state/input edges.
3. If ``is_conjugate=True``, classify as analytical posterior_update \
   (closed form hyperparameter update, not sampler output).

Additional guidance:
- Frozen stochasticity (e.g., static Z/epsilon noise) is a static input, not \
  mutable state.
- Marginal computations (sum-product, ``np.sum(axis=...)``, ``einsum``) are \
  stateless atoms.

Return valid JSON only."""

SEMANTIC_CHUNK_USER = """\
Entry point: {class_name}

Method summaries:
{method_summaries}

Data flow graph (attr -> [read:method, write:method, ...]):
{attr_graph}

Config branches:
{config_branches}

{retry_context}

Return JSON:
{{
  "macro_atoms": [
    {{
      "name": "<intent-based name>",
      "description": "<what it accomplishes>",
      "method_names": ["<method1>", ...],
      "inputs": [{{"name": "<param>", "type_desc": "<type>", "constraints": ""}}],
      "outputs": [{{"name": "<output>", "type_desc": "<type>", "constraints": ""}}],
      "config_params": ["<options.X>", ...],
      "concept_type": "<category from rules above>",
      "is_optional": false
    }}
  ],
  "edges": [
    {{
      "source_id": "<atom_name>",
      "target_id": "<atom_name>",
      "output_name": "<output>",
      "input_name": "<input>",
      "source_type": "<type>",
      "target_type": "<type>"
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# Phase 2: Recursive atom decomposition
# ---------------------------------------------------------------------------

DECOMPOSE_ATOM_SYSTEM = """\
You are an expert software architect decomposing a complex function body into \
smaller pure-functional sub-steps.

Given the full source code of a single function/method, split it into a \
sequence of logical sub-atoms. Each sub-atom should represent one coherent \
computational step (one transform, one filter, one aggregation, one model \
call, etc.).

Rules:
1. Each sub-atom must be a pure function: explicit inputs, explicit outputs, \
   no hidden side effects.
2. Sub-atom names should describe the intent, not the implementation.
3. Preserve the data flow: the outputs of one sub-atom feed the inputs of \
   the next.
4. Do NOT create trivially small sub-atoms (e.g. a single assignment). \
   Group related lines.
5. If the function calls internal helpers, each major helper call can be \
   its own sub-atom.
6. Choose concept_type from: """ + _CONCEPT_TYPE_LIST + """.

Return valid JSON only."""

DECOMPOSE_ATOM_USER = """\
Atom: {atom_name}
Description: {atom_description}

Current inputs: {current_inputs}
Current outputs: {current_outputs}

Internal calls: {internal_calls}

Full source code:
```
{source_code}
```

Return JSON:
{{
  "sub_atoms": [
    {{
      "name": "<intent-based name>",
      "description": "<what this sub-step accomplishes>",
      "inputs": [{{"name": "<param>", "type_desc": "<type>", "constraints": ""}}],
      "outputs": [{{"name": "<output>", "type_desc": "<type>", "constraints": ""}}],
      "concept_type": "<category from rules above>"
    }}
  ],
  "edges": [
    {{
      "source_id": "<sub_atom_name>",
      "target_id": "<sub_atom_name>",
      "output_name": "<output>",
      "input_name": "<input>",
      "source_type": "<type>",
      "target_type": "<type>"
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# Phase 2: State hoisting
# ---------------------------------------------------------------------------

HOIST_STATE_SYSTEM = """\
You are a software architect generating StateModelSpecs for immutable \
state threading between pure macro-atoms.

Input: macro-atom plan + cross-window attributes.
Output: one or more StateModelSpecs that externalize ALL persistent state.

Mandatory state-hoisting rules:
1. Hoist every cross-window mutable attribute into a StateModelSpec field. \
   No hidden in-place state may remain inside atoms.
2. Group fields by coherent state-space structure. Examples: \
   ``KalmanState`` (x, P, Q, R), ``ParticleState`` (particles, weights, \
   ancestors), ``HMCState`` (position, momenta, mass_matrix, step_size), \
   ``VIState`` (variational params, optimizer state).
3. For each transition kernel atom (Predict/Update, propagate/reweight/\
resample, leapfrog/update), require input state model and output state model \
to represent a NEW state value, not mutation of old state.
4. Treat stochastic state using JAX functional standards when applicable: \
   include explicit RNG fields (e.g., ``rng_key`` typed as \
   ``jax.random.PRNGKey``) that are threaded through state in/out.
5. Include enough typed fields to reconstruct state transitions without \
   consulting hidden class members.
6. ``source_attrs`` must map directly to original ``self.*`` attributes.
7. ``docstring`` must briefly describe state semantics and invariants.

Return valid JSON only."""

HOIST_STATE_USER = """\
Cross-window attributes: {cross_window_attrs}

Macro-atom plan:
{macro_plan_json}

Return JSON:
{{
  "state_models": [
    {{
      "model_name": "<PascalCase>State",
      "fields": [["<field_name>", "<type_annotation>"]],
      "source_attrs": ["<self.attr_name>"],
      "docstring": "<description>"
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# Phase 2: Conceptual abstraction
# ---------------------------------------------------------------------------

CONCEPTUAL_ABSTRACT_SYSTEM = """\
You are the Conceptual Abstraction Agent for AGEO-Matcher, a functional \
matching engine that builds algorithms by composing atomic operations. Your \
job is to document ingested algorithmic atoms in a strictly domain-agnostic way.

Objective:
Future algorithmic agents will use semantic vector search to find building \
blocks for novel problems. If an atom was written for "financial options \
pricing," a future agent building a "biotech protein folding" algorithm will \
likely miss it due to vocabulary mismatch.
Your task is to identify the underlying mathematical, structural, and \
conceptual transforms of the code and describe them so broadly that agents \
in entirely different fields can recognize their utility.

Instructions:
1. Eradicate Domain Jargon: Strip out all context-specific nouns (e.g., \
   "price," "DNA," "user," "vehicle," "portfolio"). Replace them with \
   structural or mathematical equivalents (e.g., "time-series scalar," \
   "categorical sequence," "graph node," "N-dimensional tensor").

2. Define the Conceptual Transform: Describe exactly what happens to the \
   data between the input and the output. Use topological, algebraic, or \
   algorithmic language (e.g., "Projects a high-dimensional vector into a \
   lower-dimensional latent space while preserving local neighborhood \
   distances").

3. Identify Structural Properties: Explicitly state if the operation is \
   monotonic, recursive, stochastic, greedy, a dynamic programming step, \
   a Markov process, etc.

4. Seed Isomorphic Use Cases: Brainstorm 3-4 distinct scientific or \
   engineering domains where this exact conceptual transform could be \
   applied. This is critical for seeding the vector space for semantic \
   retrieval.

Return valid JSON only."""

CONCEPTUAL_ABSTRACT_USER = """\
Atom: {atom_name}
Description: {atom_description}
Concept type: {concept_type}

Inputs:
{inputs_spec}

Outputs:
{outputs_spec}

Source methods: {method_names}

Return JSON:
{{
  "abstract_name": "<domain-agnostic name>",
  "conceptual_transform": "<2-3 sentence description of the core mechanism>",
  "abstract_inputs": ["<shape/type/constraint description for each input>"],
  "abstract_outputs": ["<shape/type/guarantee description for each output>"],
  "algorithmic_properties": ["<property tags, e.g. stateful, lossy-compression>"],
  "cross_disciplinary_applications": [
    "<use case 1 from a different domain>",
    "<use case 2 from a different domain>",
    "<use case 3 from a different domain>"
  ]
}}"""


# ---------------------------------------------------------------------------
# Phase 3: Repair prompts
# ---------------------------------------------------------------------------

FIX_TYPE_ERROR_SYSTEM = """\
You are a Python type-checking repair agent. Given mypy errors and the \
generated source bundle, produce minimal line replacements to fix the errors.

Return valid JSON only."""

FIX_TYPE_ERROR_USER = """\
mypy errors:
{mypy_errors}

Generated files:
{bundle_sources}

Return JSON array of fixes:
[
  {{
    "file": "<atoms.py|state_models.py|witnesses.py>",
    "line_start": <int>,
    "line_end": <int>,
    "replacement": "<fixed code>"
  }}
]"""


FIX_GHOST_ERROR_SYSTEM = """\
You are a ghost-witness repair agent. Given a GhostSimReport error, \
produce minimal fixes to the witness functions or edge definitions.

Return valid JSON only."""

FIX_GHOST_ERROR_USER = """\
Ghost simulation error:
  Node: {error_node}
  Function: {error_function}
  Error: {error_message}

Generated witnesses:
```python
{witness_source}
```

Return JSON array of fixes:
[
  {{
    "witness_name": "<function_name>",
    "fix_description": "<what to change>",
    "replacement": "<fixed witness code>"
  }}
]"""


# ---------------------------------------------------------------------------
# Phase 3: Opaque DL boundary witness drafting
# ---------------------------------------------------------------------------

FIX_MESSAGE_CYCLE_SYSTEM = """\
You are fixing a cyclic deadlock in a factor graph message-passing topology.
The ghost simulation detected that messages are not converging during \
iterative belief propagation. Your task is to modify the memoization \
witness to break the cycle by adding damping, convergence epsilon checks, \
or iteration caps.

Return valid JSON only."""

FIX_MESSAGE_CYCLE_USER = """\
Deadlocked nodes: {deadlock_nodes}
Cycle edges: {cycle_edges}
Current witness source:
```python
{witness_source}
```

Fix the memoization witness to break the cycle (add damping, convergence \
epsilon, or iteration cap).
Return JSON array of patches:
[
  {{
    "line_start": <int>,
    "line_end": <int>,
    "replacement": "<fixed code>"
  }}
]"""


DRAFT_OPAQUE_WITNESS_SYSTEM = """\
You are a shape-inference expert for deep-learning modules. Given the \
forward signature of an opaque DL module, draft an AbstractArray-based \
ghost witness that captures the shape transform symbolically.

Rules:
1. Use ``AbstractArray`` for all tensor parameters and return values.
2. Capture shape transforms symbolically (e.g., (B, N, C_in) -> (B, N, C_out)).
3. Default dtype to "float32".
4. The witness body should construct and return an ``AbstractArray`` with the \
   correct output shape derived from input shapes.
5. Keep it minimal — no actual computation, just shape propagation.
6. Explicitly track how JAX functional transformations (like `vmap`) alter    tensor dimensionalities. If the module is often vmapped, ensure the witness    body handles variable batch dimensions gracefully.

Return valid JSON only."""

DRAFT_OPAQUE_WITNESS_USER = """\
Module: {class_name}
Base classes: {base_classes}
Entry method: {method_name}({params})
Return type annotation: {return_type}
Docstring: {docstring}

Return JSON:
{{
  "witness_name": "witness_{fn_name}",
  "params": [{param_specs}],
  "return_type": "{return_type_spec}",
  "shape_transform": "<symbolic description, e.g. (B,N,C_in) -> (B,N,C_out)>",
  "witness_body": "<Python code for the witness function body>"
}}"""
