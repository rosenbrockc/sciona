# Sound-speed scaling correction review

Source artifact 3d49ff5d-178f-5c98-89f7-4967a10b7ff4, version
e0b55979-db6a-51ed-aafb-d3c261bc288c, remains unchanged draft. The fresh audit
checks 12 steps, 30 active bindings and 19 stored equations, with no missing
snapshot. Reviewed fields, feeds, rule IDs and all three public file pins match.

The final source mass substitution and maximum omit the factor 2 in the
square-root denominator. The consistent endpoints are
v = alpha*c*sqrt(me/(2*A*mp)) and vu = alpha*c*sqrt(me/(2*mp)). The original
endpoints exceed these by sqrt(2), as independently checked. The maximum needs
A>=1; for A=1/4 the curve exceeds the claimed maximum by a factor of two.

The source bulk-only step drops shear. Exact isotropic longitudinal speed is
sqrt((K+4G/3)/rho); its ratio to the bulk-only value is sqrt(1+4G/(3K)). Mere
K>G, as encoded by the feed, does not ensure a small correction. Step5 drops
sqrt(f), despite a feed saying it is approximately2. This changes a factor,
not a non-dominant additive term. The reconstructed scaling convention sets it
to unity and explicitly labels that approximation. Rydberg substitution for
bonding energy is likewise a model choice. Pi is a mathematical constant;
source bulk-modulus aliases are unified. Literal source AST parity is not claimed.

The [original paper](https://ccmmp.ph.qmul.ac.uk/~kostya/Speed%20of%20sound.pdf)
uses order-of-magnitude bonding energy and an omitted order-unity prefactor.
Its equations4and9 retain the denominator2. It discusses limitations for
noncohesive fluids and notes that approximations affect the numerical factor.
These physical assumptions are not established by symbolic algebra.

The proof reconstructs all12transitions, flags approximation steps1,5,7 and
independently checks substitutions, charge factorization, monotonicity,
shear correction and mass scaling.17tests pass including every corrupt step,
missing-factor regression and counterexamples to unconditional approximation
and maximum claims. Evidence: sound_scale_source_review.json and
sound_scale_proof_test_review.json.

Runtime and promotion pending. Establish a provider contract that realizes the
complete conditional scaling derivation and its corrected endpoint, retaining
the approximation diagnostics and domain restrictions. Do not approve a generic
speed helper or advertise a rigorous universal material-speed bound. Inspect
existing providers, then verify implementation, serialized graph execution,
fresh hashes and catalog serving before Tier3 promotion.


## Runtime completion (supersedes runtime-pending notes)

Implemented sound_scale with seven identically shaped inputs: A>=1, alpha,c,me,mp,
f>0 and G/K>=0. Outputs are speed_estimate, upper_scale, bulk_model_speed,
longitudinal_model_speed and shear_factor. The corrected estimate retains the
factor2. The diagnostic models retain f and shear instead of claiming they are
negligible. All models retain the Rydberg energy assumption. Eliminating a from
K=f*E/a^3 and rho=A*mp/a^3 gives the implementation directly; this is not an
unrelated sound-speed helper. No material measurement or universal-bound proof
is inferred from evaluation of these formulas.

Local450digit mpmath computes each output before float64 rounding. Positive
subnormals accepted, underflow to zero or overflow rejected. Scalar and array
inputs are copied; no broadcasting.43tests pass against a2500digit Decimal
reference reconstructing energy, density, bulk and shear moduli at synthetic
length3. This independently verifies the dimensional route and canceled length.
Tests cover sourcefactor regression, unit/bulk/shear models, invalid domains,
shape constraints, balanced extremes, subnormals and context preservation.

Six serialized CDG runner cases cover29synthetic states with0ULP discrepancies.
All five saved outputs and full graph roundtrip verified. Evidence retained in
sound_scale_test_review.json and sound_scale_execution.json. Catalog promotion
remains pending; no target has been approved from these tests alone.
