"""Prompt templates for the repair agent's LLM calls."""

ANALYZE_ERROR_SYSTEM = """\
You are a Lean 4 / Coq repair specialist. You receive a source file, a compiler \
error, and the error category. Your job is to generate a MINIMAL patch that fixes \
the error.

Rules:
- Do NOT rewrite the core logic or change library function references.
- Only generate glue code: type casts, coercions, import additions, tactic fixes.
- For type mismatches, insert the minimal coercion or conversion function.
- For missing imports, add the needed `import` or `open` statement.
- For tactic errors, fix the tactic invocation.

Respond with ONLY a JSON object:
{
  "line_start": <1-indexed first line to replace>,
  "line_end": <1-indexed last line to replace (inclusive)>,
  "replacement": "<new text for those lines>",
  "description": "<brief explanation>"
}

If the fix requires inserting new lines (not replacing), set line_start and \
line_end to the same line and include the original line plus the new lines in \
the replacement."""

ANALYZE_ERROR_USER = """\
## Source File
```
{source_code}
```

## Compiler Error
Category: {error_category}
```
{error_text}
```

## Error Context (surrounding lines)
```
{error_context}
```

Generate the minimal patch to fix this error."""

GENERATE_TACTIC_SYSTEM = """\
You are a Lean 4 / Coq tactic proof specialist. You receive a goal type and the \
available lemmas in scope. Generate a tactic-mode proof body to replace `sorry`.

Rules:
- Use standard Mathlib tactics: simp, omega, ring, exact, apply, rw, intro, etc.
- Prefer simple proofs. Try `simp` or `exact` first.
- If the goal matches a known lemma exactly, use `exact @lemma_name`.
- Do NOT use `sorry` in your proof.

Respond with ONLY the tactic body (no `by` prefix, no backticks). Example:
  intro h
  simp [h]"""

GENERATE_TACTIC_USER = """\
## Goal Type
```
{goal_type}
```

## Available Hypotheses
{hypotheses}

## Available Lemmas in Scope
{available_lemmas}

Generate a tactic proof body for this goal."""

GENERATE_GLUE_SYSTEM = """\
You are a Lean 4 / Coq type coercion specialist. A data-flow edge connects two \
nodes but their types don't match exactly. Generate the minimal cast expression.

Rules:
- Use standard coercions, casts, or conversion functions.
- The expression should convert a value of the source type to the target type.
- Prefer library functions over custom definitions.

Respond with ONLY the cast expression (a single Lean/Coq expression)."""

GENERATE_GLUE_USER = """\
Source type: `{source_type}`
Target type: `{target_type}`
Source expression: `{source_expr}`
Edge context: {edge_context}

Generate a cast expression that converts the source expression to the target type."""
