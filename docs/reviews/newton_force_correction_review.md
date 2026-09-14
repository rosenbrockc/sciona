# Newtonian force source review: inference gaps

Source a91853f4-04c2-55a3-a1d1-e47589c4b8ef, version
abf1f483-647a-5634-8ddb-45924283770e remains unchanged draft.
13steps32bindings reference18stored records:13equations and5proportionality
feeds, no missing snapshots. All reviewed fields, feed ASTs, rule IDs and
three public file hashes are checked. Feed nodes use standalone latex/sympy
fields; the validator explicitly handles their once-decoded LaTeX too.

Steps1-3 infer mass proportionality from F=ma, then rename the mass twice.
F=ma implies F proportional to m only at fixed mass-independent acceleration.
Renaming does not establish independent proportionalities of the same force to
two interacting masses. A counterexample F=m^2,a=m still satisfies F=ma.

Steps4-8 reconstruct circular force4*pi^2*m*r/T^2, requiring consistent force,
acceleration, speed, distance/circumference and time/period interpretation.
The source switches generic mass to m2 in two ASTs. Step9 multiplier uses a
wrong time identifier. Steps9-10 produce T^2*F/r=4*pi^2*m but step11 guesses
inverse-square dependence without T^2 proportional to r^3. At fixed T the
circular expression is proportional to r, giving the opposite scaling trend.
Steps12-13 combine the guessed inverse-square and unestablished independent
mass scalings, then introduce a universal G. These require physical/model
premises absent from the earlier algebra.

The retained conditional reconstruction assumes Newtonian point masses or
non-overlapping spherical bodies, positive G/masses/separation, and force
magnitude G*m1*m2/r^2. Independent checks verify mass exchange, inverse-square
scaling, test-mass acceleration, potential derivative, and a circular relation
with an explicitly supplied Kepler coefficient4*pi^2/(G*m1). The latter is a
consistency check, not a derivation of universal G from kinematics.

Eight tests pass, including all four conditional-object mutations and explicit
counterexamples. Evidence deliberately reports
source_derivation_valid_without_added_assumptions=false. It does not claim all
13source transitions are valid. A future implementation may realize the full
conditional force model while retaining these gaps and source provenance; it
must not present the original as a proved derivation. Inspect existing gravity
providers first, retain full source scope, then runner/precision/domain checks
and Tier3promotion with independent catalog verification. Runtime pending.


## Conditional runtime validated; promotion pending

Core and physics provider searches found no matching force implementation.
Added physics/newton_force.py: explicit G,mass1,mass2,separation -> force_magnitude,
acceleration_1,acceleration_2,potential_energy. The source endpoint is implemented
as an assumed constitutive law; graph metadata explicitly keeps the original
source derivation invalid without added assumptions. Both accelerations follow
F/mi and potential has zero at infinity. No circular-orbit proof or direction
vector is inferred. Body1 acceleration is due to mass2 and vice versa.

Exact rational arithmetic over copied float64 inputs gives independent rounding
without intermediate overflow. Positive inputs and identical nonempty shapes,
subnormals accepted, nonzero underflow/overflow rejected.30combinedtests pass
against2500digit Decimal potential-to-force reference, including mass exchange,
separation scaling, acceleration ordering, extremes and invalid domains.
An initial extreme fixture had unrepresentable potential energy; its expected
success was corrected. Existing rejection tests cover nonrepresentable outputs.
Six serialized runner cases cover29syntheticstates with0ULP on allfour arrays.
Graph roundtrip/freshsourceaudit pass; original gaps remain explicit.
Evidence newton_force_test_review.json and newton_force_execution.json retained.
No catalog promotion yet.
