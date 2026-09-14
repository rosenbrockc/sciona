# Orbital-radius derivation review

Source c32251b9-0a33-5ea3-be00-827caedf90c3, version
ac570cf4-7525-5ddb-8efb-63a438610c9e, remains unchanged draft.
13steps30bindings reference18equations:17stored plus recovered circumference
6785303857 from the pinned public expression file. Fresh source fields, rule IDs,
feed ASTs and all three public file hashes are checked.

The radius endpoint (G*M*T^2/(4*pi^2))^(1/3) is mathematically consistent.
Step1 incorrectly encodes the source m1 feed using the Earth target symbol;
rename the actual source pdg0005022 to pdg0005458. The force equality AST reverses
the displayed sides; equality is symmetric, but reconstruction follows the
stated centripetal=gravity orientation. Generic v and satellite v use distinct
source symbols and must denote the same speed before the equal-LHS operation.
Pi is the mathematical constant. Positive radius/period/masses/G make the
cancellations and positive cube root valid.

The physical derivation assumes uniform circular motion and an exterior
spherical gravitational field, with gravity as the sole centripetal force.
The satellite is a nonzero test mass negligible compared with the central mass.
A finite two-body relative orbit instead uses G*(M+m). Radius is measured from
the center; surface altitude and collision clearance require a body radius.

A geostationary interpretation additionally requires equatorial, prograde,
circular motion at the sidereal rotation period. Period matching alone is
insufficient. [ESA](https://www.esa.int/Enabling_Support/Space_Transportation/Types_of_orbits)
explains that direction, equatorial geometry and sidereal period distinguish
geostationary orbit. No perturbation, oblateness or station-keeping result is
implied. Generic positive periods remain valid for the conditional circular model.

The proof reconstructs all13steps and independently checks Kepler residual,
centripetal/gravity equality, inverse period and period/mass scaling.18tests pass
including corruption of every transition, an exact synthetic radius, period
convention and finite-satellite-mass counterexamples. Evidence retained in
orbit_radius_source_review.json and orbit_radius_proof_test_review.json.

Runtime pending: inspect existing providers and implement period-to-radius
realization under these exact conditions. Keep G and central mass explicit or
retain their clear mapping if using gravitational parameter; do not silently
hardcode a solar day. Validate high-precision outputs and representability,
then serialized runner/fresh evidence/Tier3promotion and independent verification.


## Runtime validated; promotion pending

No matching orbital-radius/geostationary provider was found in core/physics
provider trees. Implemented orbit_radius with three explicit inputs G, central
mass, period and two outputs radius, orbital_speed. The full period-to-radius
endpoint and its circular-speed relation are evaluated in local450digit mpmath
before independent float64 rounding. Identical shapes, positive finite inputs,
subnormal outputs allowed, nonzero underflow/overflow rejected. The domain keeps
all test-mass/circular/exterior and geostationary limitations above.

42combinedtests pass. Independent1200digit logarithmic reference uses a direct
Kepler speed expression instead of a rounded returned radius. Tests include
shape/input preservation, period scaling, extreme balanced products, subnormal
inputs, invalid domains and output range rejection. An initially misclassified
extreme had unrepresentable speed and was retained as a rejection test; another
balanced case verifies representable output despite intermediate overflow.

Six serialized runner cases cover29syntheticstates with0ULP discrepancy in both
saved arrays. Fullgraphroundtrip and freshsourceproof checked. Retained evidence
orbit_radius_test_review.json and orbit_radius_execution.json. No catalog
promotion yet; this entry supersedes runtime-pending notes only.
