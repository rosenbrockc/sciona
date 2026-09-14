# Brewster incidence angle: corrected medium ratio and angular units

Sourceca8d31c2-1220-56a1-80f4-1cd5a4f4d151, version
0c31c38a-e31c-54ec-96e4-b01d00fcff6e remains draft. Ten steps, twenty-three
bindings and fourteen stored source equations freshly checked, with exact fields,
feeds and all three public source pins. No missing source expressions.

Final source arctan(n1/n2) contradicts preceding tan(theta)=n2/n1. Correct result
is atan(n2/n1), where n1 is incident and n2 transmitted medium. Source90degrees
must become pi/2 radians; raw90 in trig AST is wrong. Angle sum/complement and
cofunction ASTs omit terms; reconstruct from LaTeX. Cofunction/tangent rename
feeds point from theta to x even though source transition requires x to theta.

Positive real indices give acute incidence/refraction, allowing division by n1
and cos(theta) and principal arctan inversion. Independently set sine/cosine of
incidence to n2/D,n1/D and refraction to n1/D,n2/D, D=sqrt(n1²+n2²). Snell and
p-polarized Fresnel numerator both vanish exactly. Interface assumed planar,
homogeneous isotropic lossless and nonmagnetic. No absorbing, anisotropic or
magnetic medium claim. S-polarized reflection is not removed.

Equal indices yield45degrees by the complementary-angle convention but reflection
vanishes at every angle, so do not call it a unique Brewster angle. Fifteen
synthetic tests pass, including ten step mutations, reversed index order,
degree/radian and arctan-branch negatives, and equal-media degeneracy.

Reference: OpenStax University Physics Volume3,1.7 Polarization:
https://openstax.org/books/university-physics-volume-3/pages/1-7-polarization

Runtime and Tier3 promotion pending. Full realization should compute incidence
and complementary refraction independently, retaining medium order and positive
index domain. Use high-precision atan2(n2,n1) and atan2(n1,n2) to avoid overflow
of ratio and cancellation from pi/2 minus an already-rounded angle. Explicitly
handle output representability near zero/grazing endpoints. Evidence must not
claim exact trigonometric equalities of rounded floating-point outputs.

Runtime complete:41tests and6runner cases29synthetic states pass, zeroULP versus
1000digit atan-ratio oracle. Independent450digit atan2 outputs avoid ratio
overflow and complement cancellation. Subnormal positive angles accepted,
underflow0 rejected. Near-grazing may round to float64pi/2; this is rounding,
not exact grazing or exact Fresnel certification. Equal indices yield pi/4
convention, no unique Brewster angle. Moderate-index Snell/Fresnel checks pass.
Source-stage pending notes are historical; catalog verification records approval.
