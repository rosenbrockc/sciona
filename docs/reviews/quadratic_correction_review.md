# Corrected real quadratic derivation — acceptable with limits

The proposed implementation solves a*x²+b*x+c=0 for both real roots under
finite real coefficients, a != 0 and b²-4ac >= 0. Its mathematical scope is
explicitly corrected from source version 40ae98f0-f033-5861-870c-6d6408edbd2b;
source parity is false. The original source and four verified arithmetic
counterexamples remain separate provenance, not erased or treated as passing.

The new proof retains b/a after division, preserves the negative and positive
branches through subtraction, and uses abs(a) when simplifying sqrt(a²).
Branches are alternatives. Output convention is lower then upper for either
sign of a, with a repeated root returned twice. An exact factorization into
a*(x-lower)*(x-upper) establishes completeness for a nonzero; negative
discriminants have no real roots and lie outside this contract. There is no
silent linear fallback or requirement that a equal b.

The implementation is numerically rearranged from the validated formula. It
classifies the discriminant using exact fractions of the converted float64
coefficients, computes its square root at 100 decimal digits, then uses a
non-cancelling q and the product identity for the second root. Exact zero and
repeated roots are handled explicitly. This scalar-per-element implementation
prioritizes accuracy; no throughput claim is made. It validates numerical
coefficients, not uncertainty in measured quantities. Consistent units across
the polynomial terms are a caller prerequisite; no physical interpretation is
inferred from generic coefficients.

Ten symbolic-proof tests reject the source defects and the incorrect omission
of abs(a). Twenty-one numerical tests exercise signs, repeated and zero roots,
shape preservation, invalid types/domains and extreme scales. Five full graph
runner cases cover 305 synthetic polynomials, both output ports and graph
serialization. The independent direct plus/minus reference uses 800-digit
Decimal arithmetic for these bounded cases; all outputs matched exactly as
float64. This does not claim correct rounding over every float64 triple.

The graph has one numerical node, three inputs and two outputs. Symbolic
transformation steps remain supporting proof evidence. The corrected mathematics and runtime behavior are acceptable with these
limitations for automated Tier 3 publication, subject to the atomic exact-version
contract, provenance and serving checks in the promotion script. No Tier 1
certification or Tier 2 usage claim is made.

Independent reference: [OpenStax Algebra and Trigonometry 2e, §2.5](https://openstax.org/books/algebra-and-trigonometry-2e/pages/2-5-quadratic-equations) derives the quadratic formula by completing the square and states the nonzero-leading-coefficient requirement. Our ordered output convention is an implementation choice; both signs describe the same solution set.
