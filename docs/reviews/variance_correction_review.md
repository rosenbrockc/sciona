# Automated review: corrected population variance

The original graph has four simplification steps and eight bindings to five
equations. Its symbolic fields are placeholders. One LaTeX intermediate changes
the square of the mean into the second moment in the negative term, invalidating
the derivation. A symmetric distribution on -1 and +1 with equal probability
has variance 1, while that intermediate produces -1. The original draft remains
unchanged; the corrected realization has a mandatory source dependency.

Fresh source review checks the five exact LaTeX pairs, source payload evidence,
symbol and inference-rule file hashes, and the source's dimensionless x identity.
The reconstruction expands the centered square, applies normalized linear
expectation with E[1]=1 and mean=E[x], preserves the square of the mean, and
simplifies to E[x²]-E[x]². It assumes a real variable with finite second moment;
no higher moments are assumed. Negative proof tests reject the faulty source
intermediate, corrupted stages and unsupported higher-degree expectations.

The numerical realization computes moments of finite nonnegative weighted
distributions. Total weight must be positive in each batch and is normalized
internally. Exact rational arithmetic on float64-converted inputs forms the
mean and centered second moment before independent float64 rounding. This
avoids subtracting large, nearly equal rounded raw moments. Variance uses the
exact mean, so a rounded output mean does not reduce variance accuracy.

Thirty provider/proof tests passed, with independent raw-moment calculations
using Decimal at 2500 digits. Six serialized CDG runner cases cover eleven
synthetic distributions with zero ULP error in both outputs. Cases include
large offsets with small spread, enormous/tiny weights, zero-weight extreme
values and subnormal variance. Inputs remain unchanged. No real dataset inputs
or identifying metadata were used.

The CDG uses dimensionless values as in the source. This is population variance
under normalized probabilities, consistent with the discrete expectation
definitions in [OpenStax Statistics](https://openstax.org/books/statistics/pages/4-2-mean-or-expected-value-and-standard-deviation).
It does not compute unbiased sample variance, standard deviation, signed-measure
moments or certified continuum quadrature. Exact zeros and subnormals are valid;
nonzero underflow to zero and overflow fail. Generic runner summaries can
overflow on extreme input magnitudes; complete computed outputs were checked.
Approval is automated Tier 3 for this corrected realization, not literal
source-AST parity or human-reviewed Tier 1 certification.
