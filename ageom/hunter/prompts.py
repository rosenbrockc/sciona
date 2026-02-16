"""Prompt templates for the Hunter agent's LLM calls."""

REFORMULATE_QUERY_SYSTEM = """\
You are a formal mathematics search expert. Given a predicate from a Predicate \
Dependency Graph and information about previous failed search attempts, generate \
new search queries to find the matching library function.

Consider:
- Alternative names for the concept (e.g., commutativity vs comm, addition vs add)
- Different levels of specificity (broader or narrower)
- Type-level reformulations
- Namespace variations (Nat.add_comm vs AddCommMonoid)

Return a JSON array of 3-5 query strings, ordered by likelihood of success.
Example: ["Nat.add_comm", "addition commutative natural", "n + m = m + n"]
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

Generate new search queries as a JSON array:
"""

SCORE_CANDIDATES_SYSTEM = """\
You are a formal mathematics expert. Given a predicate and a list of candidate \
library functions, rank them by how likely each is to be the correct match.

Consider:
- Type signature compatibility
- Semantic alignment with the predicate's informal description
- Name relevance

Return a JSON array of candidate indices (0-based), ordered from most to least \
likely. Only include candidates that have a reasonable chance of matching.
Example: [2, 0, 4]
"""

SCORE_CANDIDATES_USER = """\
## Predicate
Statement: {statement}
Description: {informal_desc}

## Candidates
{candidates_list}

Return a JSON array of indices ordered by likelihood:
"""

ANALYZE_FAILURE_SYSTEM = """\
You are a formal mathematics expert analyzing why a candidate function failed \
to type-check as a match for a predicate.

Given the compiler error output, explain:
1. Why the match failed (type mismatch, missing arguments, wrong namespace, etc.)
2. What the correct match might look like based on the error
3. Suggested search direction for the next iteration

Be concise and actionable.
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
