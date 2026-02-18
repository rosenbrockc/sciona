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
   set_theory, signal_transform, signal_filter, graph_signal_processing, custom.

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
      "concept_type": "<category>",
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
