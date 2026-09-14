# General real-angle Euler formula reconstruction

Sourcebff3440b-6913-5475-89a8-1dcc2d17c3f4, version
d37f885b-673f-5a2f-9a8b-ec0d6eebee99 remains draft. Ten source steps,
twenty-two bindings, ten stored equations and one recovered final equation
4938429483 freshly audited against all three public file pins and exact fields.
This differs from already-approved fixed Euler identity at pi, source version
36282a1e-6def-5656-a83c-1971318a6cbe.

Source y is encoded as constant Symbol but must depend on x. i/e require constant
interpretation. Source logs are encoded base10 although integration of dy/y uses
natural log. Integration loses its constant and the first integration rule says
RHS although the displayed LHS changes. Principal complex log(exp(ix))=ix is
false globally (x=2pi already disproves it). These source steps are not approved
as valid literal transformations.

Reconstruct from y=cos(x)+i*sin(x): product differentiation gives y'=i*y and
y(0)=1. Define h=exp(-ix)*y; product rule gives h'=0 on the connected real line,
with h(0)=1. Fundamental theorem of calculus gives h=1. Since exp(-ix) is nonzero,
y=exp(ix). This global real-angle proof needs no complex-log branch or division
by y, and does not assume the desired Euler formula in the differentiation.

Thirteen synthetic tests pass: certificate mutations, omitted-constant example
2exp(ix), principal-log2pi counterexample, base10 derivative mismatch and exact
angles. The proof certificate explicitly replaces invalid source integration
steps rather than claiming ten valid source algebra steps. No literal AST parity.
Runtime and Tier3 promotion pending. Preserve arbitrary real-angle scope; the
existing zero-input fixed-pi certificate is insufficient for this source.

Runtime completion: general symbolic real-angle provider returns exponential,
trigonometric expression and serialized ODE/initial-value certificate.31 tests
and6 serialized runner cases pass, including arbitrary symbolic u and u²+1/3,
exact constants and2pi logarithm counterexample. Guarded real/finite input checks
and full cache transport verified. Existing fixed-pi certificate remains separate.
Reference: MIT18.03 Differential Equations supplementary notes, complex exponential:
https://ocw.mit.edu/courses/18-03-differential-equations-spring-2010/5bd84a2e02dece25f25b5e0606b30392_MIT18_03S10_sup.pdf
Final Tier3 transaction outcome is recorded separately from historical notes.
