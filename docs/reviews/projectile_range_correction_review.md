# Projectile range: source and proof review

Status: source/proof and runtime validated; catalog promotion pending. This review
does not approve the original source graph or designate human-reviewed Tier 1.

Source version: e416bcf7-e977-5da2-b136-b9927f3236ef.
Content hash: 664efab7753078323d98d955c7c6ba727aae9622e948a31ecf1d4a5fcd5f549a.
The fresh read-only audit checks 17 steps, 40 active bindings, exact rule/feed
inventories and 23 public expressions against pinned source files. Equations
5438722682 and 9862900242 are recovered from those files; their two missing
stored bindings are not silently treated as available catalog expressions.

The reconstructed model assumes constant positive downward gravity, fixed
positive launch speed, no drag and equal launch/landing heights. Range is
horizontal displacement, not trajectory length. The angle is in radians.
Steps 2 and 9 use supplied landing boundary conditions, which do not follow
merely from identifying the final time. Step 13's feed order is reversed;
the intended substitution replaces the generic trigonometric argument with
the launch angle. The named source pi symbol is interpreted as mathematical pi.

For 0 < theta < pi/2 the nonzero flight root is 2*v*sin(theta)/g. Dividing by
flight time discards the launch root and requires this domain. At zero angle,
zero range is a continuous extension, not a separate positive-duration flight.
At pi/2 flight remains positive but horizontal range is zero.

The range is v²*sin(2*theta)/g. Its gap below v²/g is
(v²/g)*(sin(theta)-cos(theta))², a nonnegative real square. Equality on the
closed first quadrant occurs only at pi/4. This supplies a global maximum
argument beyond a stationary-point calculation. Twelve synthetic proof tests
cover certificate mutations, branch loss, unequal heights, varying speed and
uniqueness. The source ASTs are not claimed to be replayed literally.

Reference: [OpenStax University Physics 1, projectile motion](https://openstax.org/books/university-physics-volume-1/pages/4-3-projectile-motion).
The reference supports the ideal level-ground physical model; the retained
symbolic checks provide the reconstruction evidence.

Runtime: three equal-shaped arrays (launch_speed, gravity, launch_angle) return
flight_time, horizontal_range, maximizing_angle and maximum_range. The existing
projectile_trajectory atom evaluates height at supplied x and does not implement
landing or maximum-range selection, so a separate complete provider is required.
Local 450-digit evaluation precedes independent float64 rounding. Inputs are
converted/copied; nonzero output underflow and overflow reject the call.

Angles are interpreted as the actual float64 radian values. float64(pi/2) is
slightly below exact pi/2 and produces a small positive horizontal range; it is
not snapped to zero. Zero angle is explicitly the continuous extension. The
returned maximizing angle is rounded pi/4; maximum range uses the exact optimum,
not that rounded angle. This distinction is retained in the runtime documentation.

Forty synthetic proof/runtime tests pass. An independent 1200-digit Cartesian
oracle uses a complex exponential to construct velocity components, solves the
vertical landing root, and transports horizontally. Six serialized full-runner
cases cover 29 synthetic states including zero angle, near-vertical angle,
extreme scales and subnormal angle. All four saved outputs agree at zero ULP.
Graph serialization roundtrip and fresh pinned-source evidence also pass.

Next: promotion dry-run/apply, independent catalog verification and repeat-apply
idempotency check. Only automated Tier 3 publication is intended.
