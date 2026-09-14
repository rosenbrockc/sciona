# Ideal-gas isothermal compressibility: automated Tier 3 review

Source version 67defffe-886c-5f47-9d9e-e66f7bba6525, hash
49a8a2bb648c4a12b7ddbb73dab65009c7b34c66a52e7b2665367256afe6f202.

Six transitions, fourteen bindings and all eight stored source equations are
freshly reviewed. Exact LaTeX, per-step rules/bindings, pressure division feed,
source payload evidence and pinned symbol/rule/expression files are checked.
Original source remains draft and unchanged. Publication concerns the reviewed
reconstruction, not literal source AST parity.

The proof divides PV=nRT by positive P, substitutes V(P)=nRT/P into
kappa_T=-(1/V)(partial V/partial P)_T, holds temperature and amount constant,
differentiates the reciprocal, substitutes nRT=PV, and cancels to obtain 1/P.
An independent logarithmic-volume derivative confirms the endpoint. Source
derivatives of ordinary independent symbols require this functional meaning.

Source n is dimensionless even though R is molar; the reconstruction assigns
amount dimension N1. Source kappa_T is also dimensionless; the definition
requires inverse-pressure dimension M-1 L1 T2. These corrections are checked
against source pressure, temperature, volume and gas-constant dimensions.

[OpenStax College Physics 2e section 13.3](https://openstax.org/books/college-physics-2e/pages/13-3-the-ideal-gas-law)
supports PV=nRT and its molar/absolute-temperature conventions. The isothermal
compressibility here is derived from that law with fixed temperature and amount.
An isentropic path has a different derivative: the retained synthetic
V(P)=3*P^(-3/5) example yields 3/(5P), whereas the isothermal path yields 1/P.

The registered provider accepts nonempty finite positive absolute pressure
arrays in pascals, including scalar arrays. Inputs convert to float64, shape
and input values are preserved, and the exact reciprocal is rounded once.
Subnormal outputs are accepted; reciprocal overflow is rejected. The caller
establishes ideal-gas applicability. Numerical extremes test arithmetic, not
physical validity at extreme states. No real-gas, phase-transition, finite-change,
bulk-modulus, isentropic or throughput claim is made.

All 29 provider/proof tests pass, including independent Decimal precision 2500,
fixed-temperature volume paths, invalid types/values, reciprocal overflow,
six corrupted transitions and the isentropic distinction. Six serialized
full-runner cases cover 29 synthetic states with zero ULP error. Evidence binds
provider, proof, validators, tests, codec and runner. No real or templated data.

Publication scope is automated Tier 3. Fresh evidence equality, exact provider
binding, graph round trip, original-source provenance and catalog serving checks
are required before approval. No Tier 1 human-review or Tier 2 usage claim.
