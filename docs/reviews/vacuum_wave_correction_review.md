# Vacuum electric-field wave equation: source and proof review

Source version 8c5bbb6a-f2a9-50a2-8b52-1cc328beaa16, hash
08e3943f77707e8effa7b31bd8a2c64c076ace9d1baf010352e8d50c5b7deea2.

Status: source, proof, implementation and serialized execution review passed.
Automated Tier 3 publication still requires fresh evidence equality and catalog
transaction gates. The original source remains draft and unpublishable.

The six-step derivation has sixteen bindings and eleven distinct equations.
Nine stored source snapshots are freshly audited. Two missing expressions,
7575859295 and 8494839423, are recovered from the expression file pinned by
the original ingestion. Exact LaTeX pairs, per-step rule and binding identities,
the time differentiation feed and source symbol dimensions are checked.

Substantial source reconstruction is required. Several ASTs replace derivatives
with plain symbols, omit right sides, confuse vector fields with scalar symbols,
or reverse vector operands. Expression 7575859295 also has incorrect LaTeX
parentheses: the correct identity is curl(curl E)=grad(div E)-laplacian(E).
The gradient applies only to the scalar divergence. Applying it to the source's
scalar-minus-vector expression is not a meaningful vector identity.

Source E and H are dimensionless; the reconstruction assigns SI electric field
M1 L1 T-3 I-1 and magnetic field intensity I1 L-1. Source rho is mass density
M1 L-3, whereas Gauss's law here requires charge density I1 T1 L-3. Source
epsilon and mu dimensions are retained. The validator independently checks
Ampere, Faraday, Gauss and wave-equation dimensional balance after correction.
No source metadata is overwritten.

The proof uses arbitrary three-component functions E(x,y,z,t), H(x,y,z,t) and
explicit Cartesian component derivatives. Assumptions are C2 fields on a common
open space-time region, constant positive vacuum epsilon and mu, and zero charge
and current. It differentiates Ampere's equation, takes the curl of Faraday's
equation, eliminates the magnetic derivative, applies charge-free Gauss, verifies
the curl-curl identity componentwise, and cancels signs to obtain
laplacian(E)=mu*epsilon*partial_t^2(E).

[MIT 18.013A, section 29.4](https://www.ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter29/section04.html)
supports deriving the vacuum wave equation by combining Maxwell's equations,
time differentiation and the curl identity in the absence of charge and current.

All 11 proof tests pass, including arbitrary smooth components, six corrupted
transitions, an incorrect Faraday sign, a mixed polynomial vector identity, a
transverse plane wave satisfying the Maxwell premises, and a longitudinal wave
that satisfies the wave equation but violates charge-free Gauss. Consequently,
the wave equation is a necessary consequence of these Maxwell premises, not a
sufficient test of full Maxwell compliance. The work does not solve an initial
or boundary value problem. All fixtures are synthetic mathematical expressions.

The registered vacuum_wave provider takes protected srepr strings encoding
Tuple(Ex,Ey,Ez), Tuple(x,y,z,t), mu and epsilon. Coordinates must be four distinct
real Symbols. Coefficients must be provably positive and independent of all
coordinates. Conflicting same-named symbols, nonfinite expressions, non-scalar
components and constructor expression strings are rejected. The caller supplies
SI units and establishes C2 smoothness and the Maxwell premises.

Three output strings encode componentwise Laplacians, scaled second time
derivatives, and the separate divergence. Derivatives remain formal and portable
for arbitrary composed functions. They are equation sides and a constraint,
not certification of equality or Maxwell compliance. The symbolic synthesis
CDG exposes these conditions in its port contracts and regime metadata.

All 32 provider and proof tests pass. Six actual serialized runner cases include
mixed polynomials, transverse and longitudinal waves, a zero field, arbitrary
composed functions and complex field representations. Every component derivative
checks out; the longitudinal counterexample remains visible; cached AST strings
preserve type and formal derivatives. Provider, proof, validators, parser, tests,
codec and runner hashes are bound in retained evidence. No Tier 1 or Tier 2
claim is made.
