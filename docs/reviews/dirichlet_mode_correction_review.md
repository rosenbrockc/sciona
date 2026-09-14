# Normalized Dirichlet interval mode: source and proof review

Source version: 56d398db-3d6c-5144-a061-3b8c19ab598d.
Source graph hash: 2502a41c3f5c0a036b1838f3ba0d522dfba6ece9fef3e6f2919648a58ce89832.

The source has 31 steps, 76 bindings and 38 unique stored equations. Fresh
read-only validation verifies the draft graph/hash, all reviewed public fields,
inference rules, feeds, bindings and source file pins. No missing expression
records were found. This review concerns a reconstructed model, not a literal
replay or approval of the original graph.

Corrections restore blank derivative and boundary ASTs, unify coordinate and
differential aliases, replace the malformed nested normalization integral, and
evaluate the sine antiderivative at both endpoints. Normalization uses a definite
integral on [0,W], not an indefinite integral equal to one. Source steps27and28
swap the positive and negative amplitude substitutions; the reconstruction
retains both correctly paired representatives.

For W>0 and positive integer n, k=n*pi/W and
psi(x)=sqrt(2/W)*exp(i*phase)*sin(k*x). Phase is a finite real constant. The
ODE is -psi''=k²*psi in the interval interior, with psi(0)=psi(W)=0 and unit
integral of conjugate(psi)*psi. The source uses a² and conjugate(a)=a implicitly;
for a general complex amplitude normalization fixes |a|² instead. The real
positive and negative roots are phase choices, not distinct degenerate modes.
Negative integer indices likewise change only a global sign.

The zero eigenvalue requires a separate linear solution a*x+b; the two endpoint
conditions force both coefficients to zero, which cannot normalize. Negative
real eigenvalues have an exponential basis whose boundary determinant is
-2*sinh(decay*W), strictly negative for positive decay and width. The initial
check relied on SymPy to infer the sign of a difference of exponentials; it now
verifies the exponential-to-hyperbolic identity and the sign of the latter.
No spectral branch was dropped to make the check pass.

The normalized sine and boundary conditions agree with
[OpenStax's infinite-box spatial modes](https://openstax.org/books/university-physics-volume-3/pages/7-4-the-quantum-particle-in-a-box).
The present source contains no mass or Planck constant, so this realization
returns the differential-operator eigenvalue k², not quantum energy. It makes
no finite-well, arbitrary-boundary, derivative-matching, exterior distributional
extension or completeness claim. Derivative formulas apply in the interior;
endpoint values impose the stated boundary condition.

Fourteen symbolic identities verify derivatives, the eigen-equation, boundaries,
density, normalization, the definite integral, both sign representatives and
the derivative-energy integral. Separate checks exclude zero and negative
eigenvalues. Seven synthetic tests pass, including complex-phase conjugate
normalization, rejection of a noninteger mode and orthogonality of two distinct
mode examples. These checks do not validate 31 literal defective source steps.

The symbolic runtime takes width_srepr, mode_index_srepr and phase_srepr.
It preserves general symbolic parameters, rejects invalid or unproved domains,
reserves real x, and rejects coordinate dependence or conflicting assumptions.
Parsed floats become exact binary-value rationals before arithmetic. Six outputs
provide mode, opposite mode, wavenumber, first and second derivatives and a
certificate containing input ASTs, eigenvalue, density, its antiderivative and
seven checked identities. Nine ports carry explicit dimensions. Both source
signs are recovered at phase zero/pi; no extra eigenspace is claimed.

All 31 tests pass (24 runtime/contract, seven proof/scope). The independent
reference integrates the unnormalized sine to determine its normalization and
checks returned modes directly. It changes to a dimensionless integration
coordinate so large exact binary rational scale factors do not trigger costly
heuristic integration. The provider uses explicit trigonometric product-to-sum
identities and algebraic cancellation instead of heuristic trigonometric
simplification. Initial test processes were stopped after diagnosing a composite
identity failure and expensive exact-float simplification; both original cases
remain and pass. Final tests ran with an automatic 120-second process limit.

Six full serialized pipeline cases pass with exact constants, real sign choices,
complex phases, rational/irrational widths, symbolic positive integer indices and
composite parameters. Tests verify formulas, direct normalization, all certificate
identities, general parameter preservation, graph roundtrip and cached AST
strings. Retained reports bind provider, source proof/audit, tests, parser, codec
and runner. Synthetic inputs only. Automated Tier 3 promotion and independent
catalog verification are recorded separately; no Tier 1 or literal parity claim.
