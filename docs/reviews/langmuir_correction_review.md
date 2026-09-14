# Langmuir adsorption source review

Source2c931ffb-f584-5c65-bca6-ea87acd7eaaf,version
e0057db3-ea2b-5d92-b1a4-5b46572574e1 remains unchanged draft.
13steps31bindings reference19stored equations, no missing snapshots.
Fresh fields/feeds/rule identities and three public file pins checked.

Endpoint theta=K*p/(1+K*p) is correct. Step6 claims division by p*S but
produces ka/kd=B/(p*S); correct divisor is kd*p*S. Source reciprocal steps
require p>0, occupied density B>0 and vacant density S>0. At p=0 these steps
are undefined. Independently solving ka*p*S=kd*B and S+B=N for positive
ka,kd,N yields theta=ka*p/(kd+ka*p), valid also at zero pressure, where B=0
and S=N. No reciprocal replay is claimed at the boundary.

Assume a single non-dissociative adsorbate, identical independent sites,
monolayer occupancy, fixed temperature, no lateral interactions, and kinetic
equilibrium. K=ka/kd has inverse-pressure units in pressure-based kinetics:
ka has inverse-pressure inverse-time units and kd inverse-time units.
No competitive, dissociative, multilayer, transient or fitted-material claim.

All13transitions reconstructed. Independent linear solve verifies coverage,
site conservation, rate balance, zero-pressure endpoint and saturation limit;
the factored derivative ka*kd/(kd+ka*p)^2 is positive.17tests pass including
each corrupt step, missing divisor, half coverage and reciprocal boundary.
An initial monotonicity check was inconclusive before symbolic factoring;
final tests and source validation pass after that proof normalization.

Runtime pending. Inspect existing implementations, then implement complete
kinetic-to-equilibrium realization with explicit ka,kd,p and site capacity if
returning densities/rates. Preserve zero-pressure treatment and independently
computed tiny vacancy near saturation; avoid rounded1-theta cancellation.
Validate synthetic references, runner/serialization, fresh hashes and Tier3
promotion with independent serving verification. Source/proof evidence retained
in langmuir_source_review.json and langmuir_proof_test_review.json.


## Runtime complete; promotion pending

Core and physics provider searches found no existing adsorption/Langmuir
implementation. New physics/langmuir.py accepts ka,kd,p,N and returns coverage,
occupied_sites,vacant_sites,equilibrium_rate. Exact rational arithmetic over
converted float64 inputs evaluates all four outputs independently, rounding once.
The common rate is an adsorption/desorption event rate, not net accumulation.
Vacancy is computed directly, surviving saturation rounding in coverage. Zero
pressure gives exact zero coverage, occupancy and rate with vacancy N. Positive
coefficients/capacity, nonnegative pressure, identical nonempty shapes. Nonzero
underflow/overflow rejects; subnormals accepted. Single-species monolayer scope
and inverse-pressure equilibrium coefficient retained.

42combinedtests pass against independent2500digit Decimal linear-system
reference. Six serialized runner cases cover29synthetic states with0ULP on all
four saved outputs. Graph roundtrip and freshsourceaudit pass. Tests include
saturation, zero pressure, half coverage, intermediateoverflow, subnormals,
preservation and negative domains. No catalog mutation yet. Evidence retained
in langmuir_test_review.json and langmuir_execution.json.
