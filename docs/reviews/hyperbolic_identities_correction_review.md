# Hyperbolic identities: source and proof review

Status: source/proof and full symbolic runtime validated; Tier 3 promotion
pending. Original source remains draft. No human-reviewed Tier 1 designation.

Source version a6bee7b6-17cb-5f68-abe7-f56912e79dc2, content hash
3b0f152719793a679aade985da120125efa5ac123df9a110600efbb43a9b7d81.
The fresh read-only audit checks 18 steps and 43 active bindings against 24
pinned public expressions. There are 23 stored expressions; missing cosine
definition 4585932229 is recovered from the pinned expression file. Source
symbol, rule and expression files are hash-checked against snapshot pins.

The graph establishes exponential forms for sinh, cosh, tanh and sech; the
cosh²-sinh² identity; imaginary-argument sine and cosine relations; and the
sech²+tanh² identity. A runtime must cover that collection, rather than only
returning a constant for its last equation.

Use mathematical I instead of the unconstrained named source symbol for i.
Projected self-product steps reuse an equation twice although the binding set
contains it once. Step 8 supplies the equation consumed by step 7, so the
reconstruction checks step 8 first. Step 4's stored AST has already simplified
the displayed expansion to one; the original LaTex remains in source evidence.
The certificate does not claim literal uncorrected source AST replay.

For finite real dimensionless x, exp(x)+exp(-x) is strictly positive, establishing
all quotient domains. Imaginary arguments I*x are used only inside the entire
complex sine/cosine definitions. These definitions are checked directly using
exponentials, their derivative relations and initial values. The existing
real-angle Euler certificate alone would not justify complex substitution.
General complex x is excluded from the quotient runtime because cosh has zeros;
one synthetic negative example uses x=I*pi/2.

Twenty-two proof tests pass, including mutations of each of the 18 identities,
the named-i distinction, the complex pole and the zero-argument boundary.
All reconstructed equations have zero residual after exponential rewriting;
six additional checks cover the complex definitions. Replay order is
1,2,3,4,5,6,8,7,9,10,11,12,13,14,15,16,17,18.

References checked: [NIST DLMF complex trigonometric definitions](https://dlmf.nist.gov/4.14)
and [hyperbolic definitions and imaginary-argument relations](https://dlmf.nist.gov/4.28).
These support the complex definitions and domain distinction. The retained
symbolic tests provide the reconstruction evidence.

Runtime implementation now provides one argument_srepr input and three JSON
string outputs: functions_json, identities_json and certificate_json. The first
contains six unevaluated function forms; the second all18 specialized pairs;
the certificate contains generic steps, verification results and the exact
specialized outputs. General real expressions remain symbolic. The local parser
guard rejects constructor expression strings and conflicting symbol assumptions.
The certificate uses real=True for its generic argument (which implies finite),
so its serialized steps remain compatible with the guarded parser.

Forty combined tests pass, including invalid input, certificate mutation and
all-step specialization checks. Six full serialized runner cases cover zero,
negative pi/2, an exact tiny rational, sqrt(2), a real symbol and a polynomial
expression. Function forms, all18 pairs, generic certificate, graph roundtrip
and cached strings are checked. Initial validation failures came from comparing
unevaluated SymPy structure directly to evaluated expressions and rewriting
unevaluated sech(0); checks now compare serialized structure separately and
normalize only for mathematical equality. Source and runtime reports were
refreshed after the certificate serialization adjustment.

Next: Tier3 promotion dry-run/apply, independent catalog serving verification
and repeat-apply check. No numerical cancellation claim is made.
