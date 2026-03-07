"""Prompt templates for the Hunter agent's LLM calls."""

REFORMULATE_QUERY_SYSTEM = """\
You generate search queries for library matching.

Task:
- produce short, high-signal query strings
- prefer names, namespaces, aliases, and type words likely to appear in code
- avoid explanation, reasoning, markdown, and prose

Output contract:
- return ONLY a JSON array of strings
- no code fences
- no surrounding text
- 3 to 5 queries
- each query must be distinct
"""

REFORMULATE_QUERY_USER = """\
## Predicate
ID: {predicate_id}
Statement: {statement}
Description: {informal_desc}
Prover: {prover}

## Previous Queries Tried
{queries_tried}

## Compiler Errors from Last Verification
{compiler_errors}

Return ONLY the JSON array of new search queries:
"""

SCORE_CANDIDATES_SYSTEM = """\
You rank candidate library functions for matching.

Ranking rules:
- prioritize exact or near-exact type compatibility
- then prioritize semantic/name alignment with the predicate
- prefer specific domain-relevant names over generic helpers
- exclude candidates with little plausible match value

Output contract:
- return ONLY a JSON array of integer indices
- no code fences
- no prose
- no explanations
- order from most likely to least likely
- include only plausible matches
"""

SCORE_CANDIDATES_USER = """\
## Predicate
Statement: {statement}
Description: {informal_desc}

## Candidates
{candidates_list}

Return ONLY the JSON array of indices ordered by likelihood:
"""

ANALYZE_FAILURE_SYSTEM = """\
You analyze why a candidate failed verification.

Return exactly three lines in this format:
CAUSE: <short cause>
TARGET: <what the correct match likely needs>
NEXT: <best search direction>

Rules:
- be concise
- no markdown
- no extra lines
"""

ANALYZE_FAILURE_USER = """\
## Predicate
Statement: {statement}

## Failed Candidate
Name: {candidate_name}
Type: {candidate_type}

## Compiler Output
{compiler_output}

Analysis:
"""
