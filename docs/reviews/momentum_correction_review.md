# Momentum recoil automated Tier 3 review

The one-node execution CDG returns p1-p2 and its squared Euclidean norm from
real three-vectors in one orthonormal frame and consistent SI momentum units.
Recoil interpretation assumes total initial momentum p1 and final momentum
p2+recoil, with momentum conservation. The initially stationary electron model
supports these premises; energy conservation, photon dispersion, scattering
angle and event validity are outside this realization.

The original source remains draft. Its final LaTeX specifies a dot-product
identity, but the symbolic LHS is only a vector identity and RHS is blank.
The reviewed reconstruction uses explicit real 3D components. It matches seven
complete source equations and reconstructs the final squared-norm identity,
including the nonorthogonal cross term. Source input identities, duplicate
self-dot input, projected edges, symbol dimensions and rule pins are checked.
This is no literal source-AST or source inference-rule parity claim.

Provider inputs must be finite real, identical nonempty shapes (...,3), with no
broadcasting or complex values. Signed and zero components are valid. Exact
rational arithmetic computes differences and the sum of their squares before
independently rounding outputs. Squared norm therefore need not equal a norm
recomputed from the rounded recoil vector. Exact zero and subnormal outputs
are supported; nonzero values rounding to zero and nonfinite outputs fail.
Inputs are not mutated. No throughput claim.

Fifty-one tests pass, including every vector-proof component, changed premises,
missing cross term, nonorthogonal witnesses, near cancellation, large exact
cancellation, signed values, invalid inputs and output representability.
Five fresh full-runner cases cover 84 synthetic vectors; both saved outputs
exactly match independent 2500-digit Decimal arithmetic after float64 rounding.
Graph serialization round trip passes. Existing generic runner summary-statistic
reductions emit overflow warnings for extreme inputs; the validated saved
numerical outputs remain correct. No summary-statistic correctness is claimed.

Reference: OpenStax University Physics Volume 3, section 6.3,
https://openstax.org/books/university-physics-volume-3/pages/6-3-the-compton-effect
supports vector momentum conservation for an initially stationary electron.
This automated review supports Tier 3 only, with no human Tier 1 certification
or Tier 2 usage-evidence claim.
