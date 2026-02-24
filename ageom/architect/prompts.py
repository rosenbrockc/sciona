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
into smaller sub-nodes and data-flow edges between them.

Each sub-node should be a well-defined algorithmic operation. If a sub-node \
matches a known primitive from the catalog, set "is_atomic" to true and \
"matched_primitive" to the primitive name.

You must respond with ONLY a JSON object (no markdown fences, no explanation):
{{
  "sub_nodes": [
    {{
      "name": "<descriptive name>",
      "description": "<what this sub-node does>",
      "concept_type": "<paradigm category>",
      "inputs": [{{"name": "<name>", "type_desc": "<type>"}}],
      "outputs": [{{"name": "<name>", "type_desc": "<type>"}}],
      "type_signature": "<formal type if known>",
      "is_atomic": <true|false>,
      "matched_primitive": "<primitive name or null>"
    }}
  ],
  "edges": [
    {{
      "source_name": "<source sub-node name>",
      "target_name": "<target sub-node name>",
      "output_name": "<which output>",
      "input_name": "<which input>",
      "data_type": "<type of data flowing>"
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

Decompose this node into 2 or more sub-nodes with data-flow edges between them. \
Mark sub-nodes as atomic if they match a known primitive.
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
