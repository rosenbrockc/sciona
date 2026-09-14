# Time-based projectile motion: source and integration proof

Source version: 8b6af03c-7244-559f-98a1-f41d23eda79e.
Source graph hash: 3f705e752e04b90c50b873122ce1a2efedc797ea471707cd793b29747e8903bd.

The source contains 30 steps, 70 bindings and 40 unique stored expressions.
Fresh read-only validation checks the source draft state, graph hash, public
equation fields, rules, feeds, bindings and file pins. No missing records were
found. This review concerns a corrected realization, not literal AST parity or
approval of the original graph.

The source mixes instantaneous and initial vertical velocities, and one
integrated velocity AST substitutes total initial speed for a vertical component.
The sine and cosine quotient ASTs invert the ratios stated in the LaTex.
Several position, gravity, differential and velocity aliases are inconsistent.
One vector equation is blank in the AST. Treating time-dependent velocities as
independent scalar symbols would incorrectly annihilate their derivatives.
Differentials dx, dy, dv and dt must not be treated as independent positions or
velocities. A fixed Cartesian basis is required for component separation.

The reconstruction uses definite integrals from t=0, with specified initial
position and velocity, to remove ambiguity in the source's indefinite integrals.
For speed v0 and launch angle theta from positive x, it obtains
x=x0+v0*cos(theta)*t, y=y0+v0*sin(theta)*t-g*t²/2,
vx=v0*cos(theta), and vy=v0*sin(theta)-g*t. Gravity is constant and downward,
with upward-positive y. The model is an ideal point particle with no drag or
collision cutoff. Initial speed, gravity and elapsed time are nonnegative;
initial positions and angle may be signed. The zero-gravity case is the inertial
limit. Source trigonometric quotients require v0>0; the component products
independently extend to v0=0, where angle has no effect.

The scope and component equations agree with
[OpenStax University Physics Volume 1, Projectile Motion](https://openstax.org/books/university-physics-volume-1/pages/4-3-projectile-motion).
No range optimization, landing-event solver, path-length, variable-gravity or
drag model is claimed. The existing approved trajectory provider instead
eliminates time using horizontal position and requires vx0 nonzero; using it
for this graph would exclude vertical launches and fail to return the full
time-based state. A separate implementation is required.

Seventeen symbolic checks cover velocity and position integrals, derivatives,
acceleration components, all four initial conditions, speed decomposition,
zero speed, zero gravity, vertical launch and mechanical energy per unit mass.
Seven synthetic tests pass, including backward/downward launch and rejection
of a missing integration constant. The proof checks the reconstructed model;
it does not certify 30 literal uncorrected source steps.

The numerical implementation accepts initial_x, initial_y, initial_speed,
launch_angle, gravity and elapsed_time, and returns x,y,vx,vy. It preserves all
finite launch angles, zero speed/gravity/time, and both directions of motion.
Local 2000-digit intermediate calculations combine terms before independent
float64 rounding. Converted angles are literal: float64 pi/2 is not exact
vertical and is not snapped. Exact zero and subnormals are supported; nonzero
underflow or overflow in any output rejects the call. No universal rounding,
throughput, or exact rounded-energy guarantee is made.

All 29 tests pass (22 runtime/contract cases plus seven proof cases). The
independent 3000-digit reference constructs complex velocity using a complex
exponential, then integrates using endpoint-average velocity. Six serialized
pipeline cases cover 29 synthetic states with zero ULP difference on every
output. Tests include negative/downward angles, large positive and negative
angles, zero duration, zero speed, zero gravity, subnormal motion, and cancellation
that avoids intermediate float64 overflow. The graph has ten explicit SI ports
and one fully bound motion node. Retained reports bind provider, source/proof,
tests, codec and runner. Automated Tier 3 promotion and independent catalog
verification are recorded separately. No Tier 1 or literal AST parity claim.
