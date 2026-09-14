# Curl-curl Levi-Civita derivation: source and proof review

Source version 080c8bb4-b136-5809-980f-ec6ac28d6fab, hash
12d75046fb59a2f06b8f67b51f569e8fbc6104b0d4aba8f198f51636b510607c.

Status: source, reconstructed proof, implementation and serialized execution
validated. Automated Tier 3 publication requires fresh evidence equality and
catalog transaction gates. Original source remains draft and unpublishable.

The six-step source has thirteen bindings and eight distinct equations. All
eight stored snapshots are present and freshly audited. The source validator
checks exact LaTeX against the original ingestion's pinned expression file,
per-step rule/binding identities, absent feeds, source symbol declarations and
fresh payload evidence. The ASTs include incomplete expressions, nested
equalities, malformed constructors and scalar representations of vector indices.
Literal AST parity is not claimed.

The intended identity is curl(curl F)=grad(div F)-laplacian(F). The source places
the Laplacian inside the gradient, making its right side a scalar/vector mismatch.
Its second Levi-Civita factor repeats j and k incorrectly. Its delta expression
then introduces unmatched l and h indices. These are source mathematical errors,
not serialization-only differences.

In a fixed right-handed orthonormal Cartesian frame, the corrected contraction is
sum_k epsilon(i,j,k) epsilon(k,m,n) = delta(i,m)delta(j,n)-delta(i,n)delta(j,m).
All 81 combinations of i,j,m,n in {0,1,2} are checked exhaustively. The source's
double contraction over j and k instead gives 2 delta(i,n), which is also checked
as an explicit counterexample to reusing the source expression.

The source starts from the claimed identity. The reconstruction does not assume
that conclusion: it independently expands both curls of arbitrary three-component
functions, applies the exhaustively verified contraction, separates the delta
terms, performs the sums, and recovers gradient of divergence minus componentwise
Laplacian. All six stages agree componentwise. C2 smoothness justifies commuting
mixed partials. The proof uses a fixed Euclidean Cartesian basis; raising and
lowering indices therefore does not introduce metric factors.

The result applies to general C2 fields on a common open region. No charge-free,
divergence-free or Maxwell premise is used. Curvilinear coordinates, variable
metrics, nonsmooth distributions and boundary-value solutions are outside this
review. Field units are inherited through two spatial derivatives; the source's
dimensionless electric-field labels do not establish physical SI units.

All 10 proof tests pass, including six corrupted stages, a reversed contraction
sign, the invalid source double contraction and a polynomial field with nonzero
divergence. Its curl-curl is checked against explicit component values, ensuring
that dropping grad(div F) fails. Fixtures are synthetic mathematics only.

The registered curl_curl provider takes two protected AST strings encoding
Tuple(Fx,Fy,Fz) and Tuple(x,y,z). It rejects nonfinite/noncommutative fields,
incorrect tuple lengths, non-real or duplicate coordinates, conflicting symbol
assumptions, and expression strings inside constructors. All three output ASTs
retain formal derivatives, including for arbitrary composed functions. The
symbolic synthesis CDG exposes curl-curl, gradient of divergence and component
Laplacian as separate outputs with the conditional identity in their contracts.

All 27 combined implementation/proof tests pass. Six full serialized runner cases
cover explicit polynomials with nonzero divergence, transverse and longitudinal
spatial fields, constants, arbitrary composed functions and complex expressions.
Every component is checked against independent component operators, the explicit
polynomial oracle is retained, and the identity and cached string types pass.
No field is presumed divergence-free. Provider, proof, validators, tests, parser,
component operators, codec and runner are hash-bound in retained evidence.

[MIT 18.013A section 29.4](https://www.ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter29/section04.html)
uses the vector identity when deriving the wave equation. This realization
independently verifies the general contraction and does not import that
application's charge-free assumptions. No Tier 1 or Tier 2 claim is made.
