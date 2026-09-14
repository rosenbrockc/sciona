# Schwarzschild length-scale automated Tier 3 review

The implementation evaluates the complete terminal source formula R = 2GM/c²
for positive mass and explicit positive SI constants. The source graph is
version f05dd820-e273-506d-87a3-625d4cd08e76. Fresh source-algebra replay verifies
all four steps, exact expression pins and source correspondence. Its forward
squaring rule preserves principal-real-root domains and denominator bounds,
including renamed conditions; it does not infer a reverse square-root branch.

The physical reference is [OpenStax University Physics Volume 1, §13.7](https://openstax.org/books/university-physics-volume-1/pages/13-7-einsteins-theory-of-gravity),
equation 13.12. The text explicitly distinguishes the correct radius formula
from the inadequate Newtonian argument about light. This review makes that same
distinction: source-algebra parity does not establish general relativity.
The quantity is a length scale; interpreting it as a horizon requires the
Schwarzschild regime of a spherical, nonrotating, uncharged object with an
asymptotically flat vacuum exterior. The atom does not classify black holes,
model collapse, solve field equations, or locate rotating/charged horizons.

Mass is supplied in kilograms, G in m³/(kg·s²), and vacuum c in m/s. All are
finite and strictly positive. G and c must be scalar. Units and applicability
are caller responsibilities, while types, shapes and numerical bounds are
runtime checked. The scalar definitions in the pinned PDG source establish
mass, G, c and radius identities and dimensions. G and c have no implicit
hard-coded values; validation constants are explicit synthetic test inputs.

Binary mantissa/exponent arithmetic evaluates the complete formula without
avoidable intermediate overflow or underflow. Results must remain positive
and finite float64. Positive subnormal results are supported, with reduced
relative precision. Empty inputs, booleans, complex values, nonfinite values,
nonpositive values, nonscalar constants and unrepresentable results fail.
Input arrays are not mutated and output shape equals mass shape.

Twenty-five contract tests cover these boundaries, independent Decimal
arithmetic and shape witnesses. Five full graph-runner cases exercise 208
synthetic inputs through serialized graph execution, agreeing with 100-digit
Decimal calculations within one ULP (acceptance bound: three ULP). Three
positive exact rational assignments satisfy every intermediate source equality
and every inherited algebraic domain condition. The original radius and speed
in that proof are assigned the solution radius and positive c; this is an
existence check consistent with the source premise, not a universal claim about
arbitrary Newtonian escape trajectories.

The numerical realization has one executable node, three inputs and one output.
The four-step symbolic proof remains explicit supporting provenance. Acceptable
with the stated limitations for automated Tier 3 publication after exact
catalog identity, contract, binding and serving checks. No Tier 1 human
certification or Tier 2 usage claim is made.
