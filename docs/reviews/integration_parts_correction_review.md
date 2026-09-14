# Automated review: symbolic integration by parts

The source graph has three transitions and six bindings to four equations.
Its early symbolic sides do not represent the displayed differentials, and its
feed for v du is encoded as an ordinary scalar product. The reconstruction
interprets u and v as C1 functions of a real parameter on a common connected
open interval. It verifies the product rule, subtraction of v du, equality
reversal, and indefinite integration modulo an independent constant. Fresh
source checks cover four exact LaTeX pairs, source payloads, symbol identities
and dimensions, and inference-rule file pins. Original source remains unchanged.

The implementation is a symbolic synthesis transformation. It returns the
original integrand and u*v minus the unevaluated residual integral plus the
explicit integration constant. This preserves the operation described by the
source and the product-rule derivation in [OpenStax Calculus](https://openstax.org/books/calculus-volume-2/pages/3-1-integration-by-parts).
It does not silently substitute quadrature or promise a closed-form primitive.

Inputs and outputs are portable srepr AST strings. A bounded AST interpreter
accepts the reviewed mathematical grammar without Python eval. A provider-local
preflight allows string literals only in symbol/function names and numeric Float
literals, preventing implicit string-to-expression conversion inside constructors. Scalar
commutativity, a real variable, consistent symbol assumptions, independent
constant and absence of nonfinite expressions are checked. The provider verifies
the output by differentiation. Formal derivatives remain in serialized output,
avoiding internal Subs/Dummy nodes created by eager chain rules for composed
undefined functions. Returned strings are parsed again before acceptance.

Forty provider/proof tests passed, including elementary and undefined
functions, compositions, complex-valued functions of the real variable, unsafe
syntax, noncommutativity, conflicting symbols and dependent constants. Bound
dummy variables inside a definite integral used as a constant are distinguished
from free dependence. Six actual serialized CDG runner cases verify derivative
identity, residual-integral preservation and cache preservation of AST strings.
No dataset records or metadata are used.

This is conditional on the caller establishing a common C1 interval. It does
not infer poles, branches, convergence or interval applicability. The symbolic
extra declares the reviewed SymPy 1.14.0 runtime; evidence binds the parser and
provider implementation hashes. Tier 3 approval covers the reviewed symbolic
transformation, not numerical integration accuracy, literal source-AST parity,
or human-reviewed Tier 1 certification.
