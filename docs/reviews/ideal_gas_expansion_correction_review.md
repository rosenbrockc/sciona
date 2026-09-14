# Ideal-gas volumetric expansion: automated Tier 3 review

Source version 060b8ea0-d956-5fbc-9224-a31ccb1229e9, hash
000f7f4e8a1c10b86186f4c34d61bca916f2b721bf6d9f3f31d08dfeeb55bf7c.

The source has five transitions, twelve bindings and eight distinct equations.
Five equations have stored snapshots; three are recovered from the expression
file pinned by the original ingestion. The source review checks all eight
LaTeX equations, per-step rule and binding identities, the division feed,
fresh stored evidence, and pinned symbol, expression and inference files.

The source derivative placeholders need functional semantics. At fixed pressure
and amount, V(T)=nRT/P. The reconstructed first step differentiates this path
inside the definition alpha=(1/V)(partial V/partial T)_P. Subsequent steps
simplify the temperature derivative, divide PV=nRT by positive T, substitute
nR=PV/T, and cancel positive P and V to obtain alpha=1/T. An independent direct
volume-path derivative checks the endpoint.

Intermediate expression 6925244346 omits the denominator 1/(VP) in its AST,
although its LaTeX includes it. The reconstruction restores that denominator.
Source n is dimensionless while source R has molar gas constant dimensions.
The gas law requires n to have amount dimension N1. This correction is explicit;
original source metadata and ASTs remain unchanged and unapproved.

[OpenStax College Physics 2e, section 13.3](https://openstax.org/books/college-physics-2e/pages/13-3-the-ideal-gas-law)
supports the ideal gas law and its molar and absolute-temperature conventions.
The expansion coefficient here follows by differentiating that law; it is the
volumetric coefficient, not the linear coefficient sometimes denoted alpha.

The registered provider takes positive finite absolute temperatures in kelvin,
including scalar arrays, with no empty input or implicit string/bool conversion.
Exact reciprocals of converted binary64 values are rounded once. Shapes and
inputs are preserved; subnormal results are accepted and overflow rejected.
Caller establishes ideal-gas applicability at fixed pressure and amount. Extreme
numeric cases validate arithmetic only and do not establish low-temperature or
real-gas validity. No phase-transition, finite-change or throughput claim.

All 27 tests pass, covering independent Decimal precision 2500, independently
differentiated volume paths, scalar and batch shapes, extrema, invalid inputs,
reciprocal overflow, and five corrupted proof transitions. Six serialized
full-runner cases cover 29 synthetic states with zero ULP error. Evidence binds
the provider, proof, source validator, execution validator, tests, codec and
runner. No real or templated datasets are used.

Publication scope is automated Tier 3, with no Tier 1 human-review or Tier 2
usage claim. Promotion requires fresh evidence equality, exact provider version
binding, graph round trip, original-source provenance and catalog serving checks.
