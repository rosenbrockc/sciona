"""LLM prompt templates for the Smart Ingester.

Each prompt pair (SYSTEM / USER) maps to a specific phase and expected
JSON output schema.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Phase 2: Semantic chunking
# ---------------------------------------------------------------------------

SEMANTIC_CHUNK_SYSTEM = """\
You are an expert software architect specializing in converting stateful \
Python classes into stateless functional pipelines.

Given the data-flow analysis of a Python class (method summaries, self.* \
attribute graph, config-gated branches), your task is to group the methods \
into **macro-atoms** — coarse functional units named by intent, not \
implementation.

Rules:
1. Each method must appear in exactly one macro-atom.
2. Private helper methods (_foo) should be grouped with their caller.
3. __init__ preprocessing steps may form their own macro-atom or be grouped \
   with the first processing stage.
4. Config-gated branches become optional nodes (is_optional=true).
5. Name atoms by WHAT they do, not HOW: "Signal Conditioner" not "Apply Butter Filter".
6. Choose concept_type from: sorting, searching, divide_and_conquer, greedy, \
   dynamic_programming, graph_traversal, graph_optimization, string_matching, \
   geometry, arithmetic, number_theory, combinatorics, algebra, analysis, \
   set_theory, signal_transform, signal_filter, graph_signal_processing, \
   sampler, log_prob, posterior_update, variational_inference, prior_init, \
   probabilistic_oracle, mcmc_kernel, vi_elbo, sequential_filter, \
   message_passing, conjugate_update, custom.

Bayesian / probabilistic inference patterns:
- **prior_init**: Methods that initialize prior distributions, set hyperparameters, \
  or create initial parameter distributions. Look for: ``dist = Normal(...)``, \
  ``prior = Dirichlet(alpha)``, ``self.mu_0``, ``self.sigma_0``.
- **log_prob**: Methods that evaluate log-probability, log-likelihood, or score \
  functions. Look for: ``log_pdf``, ``logp``, ``log_likelihood``, \
  ``-0.5 * (x - mu)**2 / sigma**2``, ``scipy.stats.*.logpdf``.
- **sampler**: Methods that draw samples from distributions or advance MCMC chains. \
  Look for: ``np.random.``, ``rng.normal``, ``jax.random.``, Metropolis-Hastings \
  accept/reject logic (``alpha = min(1, p_new/p_old)``), HMC leapfrog steps, \
  Gibbs conditional sampling.
- **posterior_update**: Methods that perform Bayesian updates — conjugate updates \
  (``mu_n = (sigma_0^2 * x_bar + sigma^2 * mu_0) / ...``), particle filter \
  reweighting, Kalman filter update steps.
- **variational_inference**: Methods that compute ELBO, KL divergence, \
  reparameterization trick (``z = mu + sigma * eps``), or optimize variational \
  parameters.

When you detect these patterns, group them into atoms with the corresponding \
concept_type. Methods managing RNG state (seed, key splitting) should be grouped \
with the sampler atom that consumes them.

BAYESIAN EXTRACTION RULES — MANDATORY:

1. **Oracle Isolation**: If a method's metadata has ``is_oracle=True``, or if the \
   method implements a stateless log-density, gradient, or likelihood evaluation, \
   isolate it into a **pure** MacroAtomSpec with ZERO state mutation between calls. \
   The atom must have no ``writes`` to any cross-window state. Its only outputs are \
   the computed log-probability, gradient, or likelihood values. Concept type must \
   be ``log_prob`` or ``custom`` with a description stating "stateless oracle".

2. **State Decoupling**: Covariance matrices (Kalman P, mass matrices, Fisher \
   information), particle swarms (weight arrays, particle arrays), and MCMC chain \
   state (trace arrays, acceptance counters) must NEVER be hidden internal state. \
   Treat them strictly as **immutable flowing StateModelSpecs** — they enter and \
   exit atoms via explicit typed edges. If you see ``self.P``, ``self.particles``, \
   ``self.trace``, or ``self.cov`` being read AND written within the same atom, \
   you MUST split that atom so that the state flows as an input→output edge.

3. **Conjugate Short-Circuit**: If ``is_conjugate=True`` is detected on a method, \
   the atom is an analytical closed-form update. Mark its concept_type as \
   ``posterior_update`` and ensure its output is the updated hyperparameter tuple, \
   not a sample. Do not group it with stochastic samplers.

4. **Frozen Stochasticity**: If a noise matrix (Z, epsilon, eta) is initialised \
   once and reused across iterations (common in ADVI reparameterization), mark it \
   as a static initialization input to the atom — NOT as internal mutable state. \
   The atom that consumes it must declare it as a named input with \
   constraints="static standard normal noise".

5. **Factor Graph Marginals**: Methods performing ``np.sum(axis=...)`` or \
   ``einsum`` over tensor products of incoming messages are **Marginal Computation** \
   atoms. Tag them with concept_type ``custom`` and include "marginal_computation" \
   in the description. They must be stateless.

Return valid JSON only."""

SEMANTIC_CHUNK_USER = """\
Class: {class_name}

Method summaries:
{method_summaries}

Attribute graph (attr -> [read:method, write:method, ...]):
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
# Phase 2: State hoisting
# ---------------------------------------------------------------------------

HOIST_STATE_SYSTEM = """\
You are a software architect. Given a macro-atom plan and the list of \
cross-window attributes (attributes read/written across invocations), \
generate Pydantic state model specifications.

Each state model externalizes cross-window state so that atoms remain \
stateless. The graph carries state as typed edges.

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
generated source code, produce minimal line replacements to fix the errors.

Return valid JSON only."""

FIX_TYPE_ERROR_USER = """\
mypy errors:
{mypy_errors}

Generated source:
```python
{source_code}
```

Return JSON array of fixes:
[
  {{
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
