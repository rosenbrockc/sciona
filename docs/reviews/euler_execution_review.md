# Exact Euler proof realization — acceptable with limits

This realization executes a fixed exact symbolic proof of exp(I*pi)+1=0.
It has no input parameters and returns a JSON-safe certificate containing the
initial real-angle Euler identity, four computed transformations and the final
Equality(0,0). No floating-point tolerance or approximate pi is involved.

The source scope is preserved: substitute the angle with pi, simplify twice,
and add one to both sides. A single registered atom performs these four
symbolic steps and verifies the root identity by complex-exponential expansion.
It is a proof evaluator, not a general theorem prover or a physical dynamics
solver. The initial identity is checked with SymPy's established transformation
rules rather than derived from first principles.

Source scalar identities are interpreted explicitly and individually. The
imaginary-unit entry is named accordingly but marked as a variable; this
inconsistency remains in the evidence. Two integer indices with the same
printed name are not mapped. Exact pi replaces only its reviewed constant
identity. Literal source-AST parity is false; exact parity with the reviewed
interpretation is verified separately.

Ten synthetic tests cover identity mapping, changed declarations, false root
identity, exact certificate contents and independent repeated execution.
Two fresh full serialized-CDG runs each execute all four internal proof steps.
The saved certificates agree exactly with every interpreted source conclusion
and with one another. The output is a real JSON dictionary, not the runner's
fallback string representation of a symbolic object.

Evidence: euler_source_interpretation.json and euler_execution.json. The graph
contains one executable node, zero inputs and one certificate output. The realization is acceptable with the stated limitations for automated Tier 3
publication subject to exact provider identity, catalog contract and serving
checks. No Tier 1 certification or Tier 2 usage claim is made.

Independent reference: [NIST DLMF equation 4.14.3](https://dlmf.nist.gov/4.14.E3) states the exact complex exponential/trigonometric identity used as the root relation. Specializing to pi uses its exact mathematical meaning.
