# Sine double angle: automated Tier 3 review

Source version f999aa2b-2d62-57b0-8e7d-277dac1ae7a7, hash
46bf59c366a899e8b3c9d52c60661130d08a917fd7f0659d25c33151d593b50f.

Six source transitions, fourteen bindings and eight distinct equations are
reviewed. Five equations have stored snapshots; three missing distinct equations
are recovered from the expression file pinned by original ingestion. All eight
LaTeX pairs, per-step rules and bindings, feeds, source pins and fresh stored
evidence are checked. All source AST equations are then parsed and verified
under the explicit identity mapping; there is no endpoint-only source check.

The exact source imaginary-unit identity pdg0004621 is named imaginary unit
but classified as a variable. The reviewed interpretation maps it to SymPy I,
and the exact source x identity to a real dimensionless angle. The existing
identity-specific Euler semantics validator checks these pinned declarations.
This is an explicit semantic interpretation, not literal source AST parity or
general conversion of arbitrary symbols named i. Original source stays draft.

The proof establishes exponential sine/cosine premises, substitutes x -> 2x,
multiplies the two premises, scales by two, expands, cancels the opposite unit
terms and equates the common exponential right sides. Each source transition
matches the reconstructed one. Angle-addition expansion independently checks
the endpoint sin(2x)=2 sin(x) cos(x), also listed in
[OpenStax Precalculus 2e, Basic Functions and Identities](https://openstax.org/books/precalculus-2e/pages/a-basic-functions-and-identities).

The registered runtime provider takes nonempty finite real radians, including
scalar arrays, converted to float64. It evaluates the product with an isolated
450-decimal-digit mpmath context, avoiding binary64 angle doubling and argument
clipping. The precision optional dependency declares mpmath 1.3.0. Inputs and
shape are preserved, subnormal outputs supported, and nonzero underflow to zero
rejected. Accuracy is evidenced for the corpus, not universally proved correct
rounding. Complex inputs, degrees, phase unwrapping, interval arithmetic and
throughput are outside the contract.

All 31 tests pass. They cover an independent direct sin(2x) oracle at 1000
decimal digits, large positive and negative arguments through float64 extrema,
near-root and subnormal inputs, invalid values and types, concurrent precision
context isolation, six corrupted transitions and unknown source identities.
Six serialized full-runner cases cover 29 synthetic angles with zero ULP error.
Evidence binds provider, proof, source and execution validators, tests, source
parser/constant semantics, codec, runner and dependency declaration. No real or
templated dataset contents are used.

Automated Tier 3 is the publication scope. There is no Tier 1 human-review or
Tier 2 usage claim. Publication gates require fresh evidence equality, exact
provider binding, graph round trip, source provenance and verified catalog serving.
