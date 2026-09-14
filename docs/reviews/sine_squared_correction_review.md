# Sine-squared identity source review

Source c48b9b6a-d0fb-56e3-a2e7-441ddd5e84d3, version
240e8f71-cd8a-5890-99a4-54d082f4a31d, remains unchanged draft.
13steps28bindings reference15distinct equations;13stored and2recovered from
pinned public expressions:4938429483(Euler premise),9988949211(endpoint).
The three missing binding occurrences correspond to those two equations.
Fresh source fields, rule IDs, feed ASTs and three public file hashes match.

The endpoint sin(x)^2=(1-cos(2x))/2 is correct. Reconstruction treats i as the
imaginary unit and x as real dimensionless radians. Source7572664728 suddenly
uses the position symbolpdg0004037 rather than the anglepdg0001464. Restore the
angle consistently. Step1 feeds must rename x to2x rather than the reverse.
The source step2 consumes an expansion produced by step4: execute in order
1,3,4,2,5,6,7,8,9,10,11,12,13. Multiplying Euler by itself requires both copies
of the premise despite a single distinct binding identity.

The separately checked Euler ODE proof is a prerequisite, with its source code
hash retained. The Pythagorean premise is checked through its zero derivative
and value1at0. All13transitions are reconstructed, including real-part selection
and the intermediate rearrangements. An independent endpoint certificate checks
that both sides satisfy y third derivative+4*y first derivative=0 and initial
values y(0)=0,y first derivative(0)=0,y second derivative(0)=2. Linear ODE
uniqueness establishes equality without invoking a half-angle simplifier.

19tests pass, including each corrupt step, four exact angles, and a negative-sine
example excluding a principal-square-root interpretation. Runtime pending.
Inspect existing providers before choosing a symbolic or numeric realization;
retain the squared identity and proof scope. If numeric, avoid subtracting a
rounded cosine from1 near zeros. Require independent execution/serialization
and fresh hash checks before Tier3promotion; no sign-of-sine inference.


## Symbolic runtime complete; promotion pending

Implemented physics/sine_squared.py with guarded finite-real-angle srepr input
and three strings: sine_squared_srepr, half_angle_srepr, certificate_json.
The implementation retains all13source transitions, Euler/Pythagorean checks,
and independent endpoint differential certificate. Local parser guards copied
from the reviewed symbolic providers prevent expression strings in constructors.
Conflicting symbol assumptions are rejected. No dependency on another provider
at runtime; prerequisite checks are self-contained. Requires symbolic extra.

37proof/runtime tests pass. Six complete serialized runner cases include zero,
negative pi/2, exact tiny rational, sqrt2, a general real symbol and a composite
real expression. Both ASTs, certificate, dependency order, graph round trip and
cache strings are verified. Output expressions remain symbolic; no floating-point
cosine subtraction or sine-sign selection is performed. Consumers evaluating the
half-angle form numerically must choose suitable precision themselves.

Retained evidence: sine_squared_test_review.json and sine_squared_execution.json.
Tier3promotion has not yet run; original source remains draft.
