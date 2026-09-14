# Automated review: normalized Hermitian expectation

Source graph c9b2afed-ec6a-569a-9a39-e74086c8d737 has four adjoint/substitution
steps and ten bindings to six equations. Most stored symbolic sides are empty
or placeholders, so they cannot certify the original graph. Fresh evidence
checks each bound source payload, exact reviewed LaTeX sides, symbol-file pin
and inference-rule pin. The original graph remains unchanged and unapproved.

The reconstruction uses arbitrary positive finite dimension, an operator A,
a complex column psi and an initially unconstrained complex scalar q. Explicit
premises are A†=A and psi†Apsi=q. It takes adjoints on both sides, distributes
them in reverse order, substitutes Hermiticity and recovers conjugate(q)=q.
The proof verifies these four transitions without assuming q is real. Negative
tests reject corrupted steps, a missing Hermitian premise and ordinary transpose
substituted for the adjoint. These identities agree with the adjoint treatment
in [MIT Quantum Physics II notes](https://ocw.mit.edu/courses/8-05-quantum-physics-ii-fall-2013/4de6d044fa9d7e5b8998c5f8ca984a42_MIT8_05F13_Chap_04.pdf).

The numerical realization makes state normalization explicit by dividing the
real quadratic form by psi†psi, which is strictly positive for every admitted
nonzero state. Operator and state use a common orthonormal basis. Inputs convert
to complex128; exactly Hermitian matrices are required without tolerance or
symmetrization. Matching batches do not broadcast. Exact rational arithmetic
on the converted components prevents intermediate overflow or loss from state
normalization; the final ratio rounds once to float64. Exact zero and subnormal
outputs are admitted; nonzero underflow to zero and overflow are rejected.

Thirty-one provider tests and seven proof tests passed. Five serialized CDG
runner cases cover twelve states, including complex batches, imaginary
off-diagonal signs, enormous cancelling terms and subnormal scales. Results
match independent mpmath complex matrix multiplication at 2500 decimal digits
with zero ULP error. All fixtures are purely synthetic mathematical inputs.

This is finite-dimensional pure-state algebra. It makes no claim about domains
of unbounded infinite-dimensional operators, measurement sampling, or execution
throughput. The generic runner's summary statistics cast complex extrema/means
to real and may overflow on extreme input magnitudes; saved complex input arrays
and the checked numerical outputs retain their values. That diagnostic limitation
is recorded rather than mistaken for loss in the computation. The realization
is an explicit source reconstruction suitable for automated Tier 3 review, not
literal source-AST parity or human-reviewed Tier 1 certification.
