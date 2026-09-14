# Hermitian eigenstate overlap source reconstruction

Source graph 910ed367-b927-5e0b-a58a-2c6ed28222fe; version
b46294cb-9225-5f6d-b945-1e30d4770bc8; original remains draft.

The fresh read-only audit verifies seven inference steps, fifteen active bindings,
eight stored equations and all three public file pins. It compares exact stored
symbolic fields and reviewed LaTeX; public Cypher doubles some LaTeX escapes,
which ingestion decodes once. This is a reconstruction, not source AST parity.

The source represents an operator as a commuting scalar multiplied by Bra/Ket
functions. Replace that with ordered finite-dimensional matrix products. Two
intermediate equations have empty symbolic right sides; reconstruct them from
LaTeX. The equality of the beta and alpha matrix elements incorrectly repeats
beta in the stored AST; the right side must use alpha. The variable x is a matrix
element, not a spatial coordinate. All operator/eigenvalue/matrix-element values
share an arbitrary operator unit; discrete vector components are dimensionless.

Assume A Hermitian, nonzero right eigenvectors u and v, with real eigenvalues
a and b. Obtain the left eigenvector equation by taking the adjoint of A u=a u
and using A†=A. Compute u† A v in both ways; subtract and factor to obtain
(b-a)u†v=0. The source stops at this product identity. Orthogonality follows only
for distinct eigenvalues. Equal eigenvalues permit nonzero overlap. No assertion
about unbounded operators or infinite-dimensional domains is made.

The matrix proof checks all seven reconstructed steps for arbitrary positive
finite dimension. Fifteen synthetic tests pass, including mutations of every
step, missing premises, a complex Hermitian example that catches ordinary
transpose, and counterexamples for degenerate/non-Hermitian/non-eigenstate cases.

Reference: MIT 8.321 Quantum Theory I, Lecture 2,
https://ocw.mit.edu/courses/8-321-quantum-theory-i-fall-2017/36c32531fa12a2d99687cb3f0ac58502_MIT8_321F17_lec2.pdf

Runtime provider and six serialized runner cases now pass. Combined provider/proof
suite: 41 tests. Exact scalar-sum oracle verifies overlap, matrix element and
computed gap product. Degenerate cases preserve complex nonzero overlap, including
unnormalized vectors; zero operators and symbolic real common eigenvalues covered.
Provider verifies finite scalar entries, Hermiticity, nonzero states and both
eigenvector equations by exact simplification. Undecidable premises are rejected.
Guarded constructor parsing rejects embedded expression strings. Round-trip graph
serialization and cached AST strings verified. Tier 3 transaction pending.
The existing hermitian_expectation provider covers a single quadratic form and
does not establish this two-eigenstate relation.
