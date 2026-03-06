"""Prompt templates for the decomposition engine.

All follow the Hunter pattern: system/user string constants with .format(),
expecting JSON output from the LLM.
"""

# ---------------------------------------------------------------------------
# 1. SELECT_STRATEGY — pick algorithmic paradigm
# ---------------------------------------------------------------------------

SELECT_STRATEGY_SYSTEM = """\
You are an expert algorithm designer. Given a high-level algorithmic goal, \
select the best algorithmic paradigm for decomposing it.

You must respond with ONLY a JSON object (no markdown fences, no explanation):
{{"paradigm": "<paradigm_value>", "rationale": "<why this paradigm>", "variant_hint": "<specific variant if applicable>"}}

The "paradigm" field must be one of these exact values:
{available_paradigms}
"""

SELECT_STRATEGY_USER = """\
Goal: {goal}

Select the best paradigm for decomposing this goal into sub-problems. \
Consider the structure of the problem and which paradigm's skeleton best fits.
"""

# ---------------------------------------------------------------------------
# 2. DECOMPOSE_NODE — break a node into sub-nodes and edges
# ---------------------------------------------------------------------------

DECOMPOSE_NODE_SYSTEM = """\
You are an expert algorithm designer. Decompose the given algorithmic node \
into conceptual sub-nodes and lightweight ordering hints.

IMPORTANT:
- Focus only on conceptual structure.
- Do NOT emit inputs, outputs, type signatures, glue code, or typed edges.
- Do NOT try to solve the entire implementation. Propose the minimum useful set of conceptual steps.
- Deterministic tooling will synthesize IO ports, atomic checks, and edge types.
- If a step clearly matches a known primitive, include `matched_primitive_hint`.

You must respond with ONLY a JSON object (no markdown fences, no explanation):
{{
  "progress_updates": [
    "<high-level checkpoint 1>",
    "<high-level checkpoint 2>"
  ],
  "sub_nodes": [
    {{
      "name": "<descriptive name>",
      "description": "<what this conceptual step does>",
      "concept_type": "<optional paradigm category>",
      "matched_primitive_hint": "<optional primitive name hint>"
    }}
  ],
  "flow_hints": [
    {{
      "from": "<source sub-node name>",
      "to": "<target sub-node name>",
      "why": "<brief ordering/dependency rationale>"
    }}
  ]
}}
"""

DECOMPOSE_NODE_USER = """\
Node to decompose:
  Name: {node_name}
  Description: {node_description}
  Concept type: {concept_type}
  Inputs: {inputs}
  Outputs: {outputs}
  Current depth: {depth}
  Max depth: {max_depth}

Relevant primitives from the catalog:
{primitives}

{example_decompositions}

{retry_context}

Decompose this node into 2 or more conceptual sub-nodes.
Only describe what each step is responsible for.
Include optional flow_hints for ordering/dependencies.
Do not emit `inputs`, `outputs`, `type_signature`, `is_atomic`, or explicit `edges`.
Also include 2-6 concise `progress_updates` describing major decomposition checkpoints.
"""

# ---------------------------------------------------------------------------
# 3. CRITIQUE — validate a decomposition
# ---------------------------------------------------------------------------

CRITIQUE_SYSTEM = """\
You are an expert algorithm critic. Evaluate whether a proposed decomposition \
of an algorithmic node is correct, complete, and well-structured.

Check for:
1. Semantic completeness — do the sub-nodes fully implement the parent?
2. Type compatibility — do the data-flow edges have compatible types?
3. No missing steps — is there a clear path from inputs to outputs?
4. Appropriate atomicity — are atomic claims justified by the catalog?

You must respond with ONLY a JSON object (no markdown fences, no explanation):
{{
  "approved": <true|false>,
  "reason": "<explanation>",
  "io_issues": ["<issue1>", "<issue2>"],
  "flagged_nodes": ["<node_name that needs attention>"]
}}
"""

CRITIQUE_USER = """\
Parent node:
  Name: {parent_name}
  Description: {parent_description}
  Inputs: {parent_inputs}
  Outputs: {parent_outputs}

Proposed sub-nodes:
{sub_nodes}

Proposed edges:
{edges}

Depth constraints: current={current_depth}, max={max_depth}

Relevant primitives from the catalog:
{primitives}

Evaluate this decomposition. Approve it only if it is correct and complete.
"""
